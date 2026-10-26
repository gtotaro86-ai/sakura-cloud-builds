"""Strict bounded HTTP Range inspection of immutable native GGUF sources.

Never falls back to a full download when a server ignores Range. Metadata inspection uses at most the header budget. A bounded 1 MiB
prefetch can overlap the beginning of the first tensor payload.
"""
from __future__ import annotations
import re
import struct
import urllib.parse
import urllib.request
from dataclasses import dataclass
from hfqa.gguf import GGUFError, TENSOR_LAYOUTS

MAX_RANGE_BYTES = 8 * 1024**2
MAX_HEADER_BYTES = 64 * 1024**2


def descriptor_url(descriptor: dict) -> str:
    repo, rev, name = descriptor['repo_id'], descriptor['revision'], descriptor['filename']
    if not re.fullmatch(r'[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)?', repo) or any(part in ('.','..') for part in repo.split('/')) or not re.fullmatch(r'[a-f0-9]{40}', rev):
        raise ValueError('source must be an immutable Hugging Face model revision')
    if name.startswith('/') or any(part in ('','.','..') for part in name.split('/')) or any(ord(c)<32 or ord(c)==127 for c in name) or '\\' in name or ':' in name or not name.endswith('.gguf'):
        raise ValueError('invalid remote GGUF filename')
    return f'https://huggingface.co/{urllib.parse.quote(repo,safe="/")}/resolve/{rev}/{urllib.parse.quote(name,safe="/")}'


class RangeReader:
    def __init__(self, descriptor: dict, opener=None):
        self.url=descriptor_url(descriptor)
        self.size=int(descriptor['size'])
        if isinstance(descriptor['size'],bool) or self.size<24: raise ValueError('invalid GGUF source size')
        self.opener=opener or urllib.request.urlopen
        self.bytes_read=0
        self.request_count=0

    def read_range(self, offset: int, count: int) -> bytes:
        if type(offset) is not int or type(count) is not int or offset<0 or count<0 or count>MAX_RANGE_BYTES or offset+count>self.size:
            raise ValueError('range outside source or bounded read limit')
        if count==0:return b''
        end=offset+count-1
        req=urllib.request.Request(self.url,headers={'Range':f'bytes={offset}-{end}', 'Accept-Encoding':'identity', 'User-Agent':'hf-quant-agent/0.2'})
        with self.opener(req,timeout=45) as response:
            expected=f'bytes {offset}-{end}/{self.size}'
            if response.status!=206 or response.headers.get('Content-Range')!=expected:
                raise IOError('server did not honor the exact pinned HTTP Range; full download refused')
            if response.headers.get('Content-Encoding', 'identity') not in ('identity',''):
                raise IOError('compressed range response rejected')
            blob=response.read(count+1)
        if len(blob)<count:
            from hfqa.retries import TransientTransferError
            raise TransientTransferError('truncated range response')
        if len(blob)>count:raise IOError('oversized range response')
        self.bytes_read+=len(blob);self.request_count+=1
        return blob

    def iter_range(self, offset: int, count: int, chunk_size: int=MAX_RANGE_BYTES):
        if type(offset) is not int or type(count) is not int or offset<0 or count<0 or offset+count>self.size:raise ValueError('invalid source interval')
        if type(chunk_size) is not int or chunk_size<=0 or chunk_size>MAX_RANGE_BYTES:raise ValueError('invalid chunk size')
        while count:
            size=min(count,chunk_size)
            yield self.read_range(offset,size)
            offset+=size;count-=size


@dataclass(frozen=True)
class RemoteTensor:
    name:str
    shape:tuple[int,...]
    type:int
    offset:int
    nbytes:int


@dataclass
class RemoteGGUF:
    version:int
    alignment:int
    metadata:dict
    metadata_entries:list[tuple[str,int,bytes]]
    tensors:list[RemoteTensor]
    data_offset:int
    reader:RangeReader


class _Header:
    def __init__(self, reader, limit):
        self.reader=reader;self.limit=min(limit,reader.size)
        self.pos=0;self.data=bytearray()

    def read(self,n):
        if n<0 or self.pos+n>self.limit:raise GGUFError('GGUF header exceeds bounded metadata budget')
        while len(self.data)<self.pos+n:
            count=min(1024**2,self.limit-len(self.data))
            self.data.extend(self.reader.read_range(len(self.data),count))
        result=bytes(self.data[self.pos:self.pos+n]);self.pos+=n
        return result

    def u32(self):return struct.unpack('<I',self.read(4))[0]
    def u64(self):return struct.unpack('<Q',self.read(8))[0]
    def string(self):
        count=self.u64()
        if count>8*1024**2:raise GGUFError('GGUF string exceeds metadata budget')
        return self.read(count).decode('utf-8',errors='strict')

    def value(self, kind, depth=0, decode=True):
        if depth>4:raise GGUFError('GGUF metadata nesting exceeds budget')
        formats={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
        if kind in formats:
            fmt='<'+formats[kind];return struct.unpack(fmt,self.read(struct.calcsize(fmt)))[0]
        if kind==8:return self.string()
        if kind==9:
            subtype=self.u32();count=self.u64()
            if count>2_000_000 or subtype not in {*formats,8}:raise GGUFError('GGUF metadata array exceeds budget or has invalid type')
            if subtype in formats:
                raw=self.read(count*struct.calcsize('<'+formats[subtype]))
                if count<=4096:return list(struct.unpack('<'+formats[subtype]*count,raw))
            else:
                for _ in range(count):self.value(subtype,depth+1,False)
            return None
        raise GGUFError(f'unsupported metadata type {kind}')


def inspect_remote_gguf(descriptor: dict, *, reader=None, max_header_bytes=MAX_HEADER_BYTES, allow_split=False, extra_layouts=None) -> RemoteGGUF:
    """`extra_layouts` names tensor formats of a modified runtime ({type id: (elements per block, bytes per block)}) that the caller can
    handle; they are checked against the file size like every stock format."""
    reader=reader or RangeReader(descriptor)
    layouts={**TENSOR_LAYOUTS,**(extra_layouts or {})}
    h=_Header(reader,max_header_bytes)
    if h.read(4)!=b'GGUF':raise GGUFError('invalid remote GGUF magic')
    version=h.u32()
    if version not in (2,3):raise GGUFError('unsupported remote GGUF version')
    count,kv_count=h.u64(),h.u64()
    if count>1_000_000 or kv_count>100_000:raise GGUFError('unreasonable GGUF tensor or metadata count')
    entries=[];metadata={}
    for _ in range(kv_count):
        key=h.string()
        if key in metadata:raise GGUFError('duplicate GGUF metadata key')
        kind=h.u32();start=h.pos;value=h.value(kind)
        entries.append((key,kind,bytes(h.data[start:h.pos])));metadata[key]=value
    alignment=metadata.get('general.alignment',32)
    if type(alignment) is not int or alignment<1 or alignment>4096 or alignment&(alignment-1):
        raise GGUFError('invalid GGUF alignment')
    if count==0 and not (allow_split and type(metadata.get('split.count')) is int and metadata['split.count']>1):
        raise GGUFError('unreasonable GGUF tensor or metadata count')  # only a metadata shard of a split set may hold no tensors
    arch=metadata.get('general.architecture')
    secondary=allow_split and type(metadata.get('split.no')) is int and metadata['split.no']>0
    if not secondary and (not isinstance(arch,str) or not re.fullmatch('[a-z0-9_-]+',arch)):raise GGUFError('GGUF source has no valid architecture')
    if not allow_split and int(metadata.get('split.count') or 1)>1:raise GGUFError('split GGUF input requires the multi-source streaming adapter')
    tensors=[];names=set();ranges=[]
    for _ in range(count):
        name=h.string()
        if not name or '\x00' in name or name in names:raise GGUFError('empty, invalid or duplicate remote tensor name')
        names.add(name);ndim=h.u32()
        if ndim<1 or ndim>4:raise GGUFError('invalid tensor dimensions')
        shape=tuple(h.u64() for _ in range(ndim));kind=h.u32();offset=h.u64()
        if any(x<=0 or x>2**40 for x in shape) or kind not in layouts or offset%alignment:
            raise GGUFError('unsupported or invalid remote tensor layout')
        block,width=layouts[kind]
        if shape[0]%block:raise GGUFError('tensor row does not fit native blocks')
        elements=1
        for n in shape:elements*=n
        nbytes=elements//block*width
        tensors.append(RemoteTensor(name,shape,kind,offset,nbytes));ranges.append((offset,offset+nbytes))
    data_offset=(h.pos+alignment-1)//alignment*alignment
    previous=0
    for start,end in sorted(ranges):
        if start<previous or data_offset+end>reader.size:raise GGUFError('overlapping or truncated remote tensor payload')
        previous=end
    return RemoteGGUF(version,alignment,metadata,entries,tensors,data_offset,reader)


class ConcatenatedRangeReader:
    """Virtual source concatenation; verifies each original shard independently."""
    def __init__(self, descriptors, readers=None):
        import bisect
        if not descriptors or len(descriptors)>1024:raise ValueError('invalid source set size')
        self.descriptors=list(descriptors)
        self.readers=list(readers) if readers is not None else [RangeReader(d) for d in descriptors]
        if len(self.readers)!=len(descriptors):raise ValueError('source reader count mismatch')
        self.starts=[];self.size=0
        for desc,reader in zip(descriptors,self.readers):
            if reader.size!=desc['size'] or not re.fullmatch('[a-fA-F0-9]{64}',desc.get('sha256','')):raise ValueError('invalid pinned shard identity')
            self.starts.append(self.size);self.size+=reader.size
        self._hashes=None;self._cursor=0;self._last_reader=None

    def begin_verified_pass(self):
        import hashlib
        self._hashes=[hashlib.sha256() for _ in self.readers];self._cursor=0;self._last_reader=None
        for reader in self.readers:
            if hasattr(reader,'clear_cache'):reader.clear_cache()

    def read_range(self,offset,count):
        import bisect
        if type(offset) is not int or type(count) is not int or offset<0 or count<0 or count>MAX_RANGE_BYTES or offset+count>self.size:raise ValueError('invalid virtual source range')
        if self._hashes is not None and offset!=self._cursor:raise ValueError('verified source pass must be contiguous')
        result=bytearray()
        while count:
            index=bisect.bisect_right(self.starts,offset)-1
            if self._last_reader is not None and self._last_reader!=index:
                previous=self.readers[self._last_reader]
                if hasattr(previous,'clear_cache'):previous.clear_cache()
            self._last_reader=index
            local=offset-self.starts[index];size=min(count,self.readers[index].size-local)
            raw=self.readers[index].read_range(local,size)
            if len(raw)!=size:raise IOError('short source shard range')
            if self._hashes is not None:self._hashes[index].update(raw);self._cursor+=size
            result.extend(raw);offset+=size;count-=size
        return bytes(result)

    def finish_verified_pass(self):
        if self._hashes is None or self._cursor!=self.size:raise ValueError('source verification pass is incomplete')
        proofs=[]
        for desc,digest in zip(self.descriptors,self._hashes):
            if digest.hexdigest()!=desc['sha256'].lower():raise ValueError('original GGUF source shard SHA-256 mismatch')
            proofs.append({**desc,'sha256_verified':True})
        self._hashes=None
        return {'sources_verified':proofs,'total_source_bytes':self.size,'all_source_shas_verified':True}


def inspect_remote_source_set(descriptors: list[dict], *, readers=None) -> RemoteGGUF:
    if not descriptors or len(descriptors)>1024:raise GGUFError('invalid GGUF source set size')
    if readers is not None and len(readers)!=len(descriptors):raise GGUFError('source reader set incomplete')
    ordered=sorted(zip(descriptors,readers or [None]*len(descriptors)),key=lambda pair:pair[0].get('split_index',1))
    descriptors=[pair[0] for pair in ordered]
    if any(type(d.get('split_count',1)) is not int or d.get('split_count',1)!=len(descriptors) for d in descriptors):raise GGUFError('incomplete split source set')
    if any(type(d.get('split_index',1)) is not int for d in descriptors) or [d.get('split_index',1) for d in descriptors]!=list(range(1,len(descriptors)+1)):raise GGUFError('split source order is incomplete')
    identities={(d['repo_id'],d['revision']) for d in descriptors}
    if len(identities)!=1 or len({d['filename'] for d in descriptors})!=len(descriptors):raise GGUFError('source shards must have unique filenames in one pinned repository revision')
    actual_readers=[pair[1] or RangeReader(pair[0]) for pair in ordered]
    first=None;all_tensors=[];names=set();base=0;declared_totals=[]
    for index,(desc,reader) in enumerate(zip(descriptors,actual_readers)):
        part=inspect_remote_gguf(desc,reader=reader,allow_split=True)
        if hasattr(reader,'clear_cache'):reader.clear_cache()
        if first is None:first=part
        if len(descriptors)==1 and not any(k.startswith('split.') for k in part.metadata):return part
        if type(part.metadata.get('split.no')) is not int or type(part.metadata.get('split.count')) is not int or part.metadata.get('split.no')!=index or part.metadata.get('split.count')!=len(descriptors):raise GGUFError('split header disagrees with immutable file set')
        total=part.metadata.get('split.tensors.count')
        if type(total) is not int or total<1:raise GGUFError('split tensor completeness metadata missing')
        declared_totals.append(total)
        if part.metadata.get('general.architecture',first.metadata['general.architecture'])!=first.metadata['general.architecture']:raise GGUFError('split architecture mismatch')
        for tensor in part.tensors:
            if tensor.name in names:raise GGUFError('duplicate tensor across source shards')
            if len(all_tensors)>=100000:raise GGUFError('complete tensor catalogue exceeds bounded planner budget')
            names.add(tensor.name)
            all_tensors.append(RemoteTensor(tensor.name,tensor.shape,tensor.type,base+part.data_offset+tensor.offset,tensor.nbytes))
        base+=reader.size
    if any(total!=len(all_tensors) for total in declared_totals):raise GGUFError('split tensor completeness check failed')
    metadata={k:v for k,v in first.metadata.items() if not k.startswith('split.')}
    entries=[e for e in first.metadata_entries if not e[0].startswith('split.')]
    virtual=ConcatenatedRangeReader(descriptors,actual_readers)
    return RemoteGGUF(first.version,first.alignment,metadata,entries,all_tensors,0,virtual)


# Worker entry point retained as a descriptive public alias.
inspect_remote_gguf_sources = inspect_remote_source_set

class BufferedRangeReader:
    """One bounded HTTP window for sequential tensor reads, with no disk cache."""
    def __init__(self, reader, window_bytes=MAX_RANGE_BYTES):
        if type(window_bytes) is not int or window_bytes<1 or window_bytes>MAX_RANGE_BYTES:raise ValueError('invalid range cache size')
        self.inner=reader;self.size=reader.size;self.window_bytes=window_bytes
        self._start=-1;self._cache=b''

    def clear_cache(self):
        self._start=-1;self._cache=b''

    @property
    def bytes_read(self):return self.inner.bytes_read

    @property
    def request_count(self):return self.inner.request_count

    def read_range(self,offset,count):
        if type(offset) is not int or type(count) is not int or offset<0 or count<0 or count>MAX_RANGE_BYTES or offset+count>self.size:raise ValueError('invalid buffered source range')
        result=bytearray()
        while count:
            if not self._start<=offset<self._start+len(self._cache):
                self._start=offset//self.window_bytes*self.window_bytes
                self._cache=self.inner.read_range(self._start,min(self.window_bytes,self.size-self._start))
            local=offset-self._start;size=min(count,len(self._cache)-local)
            result.extend(self._cache[local:local+size]);offset+=size;count-=size
        return bytes(result)
