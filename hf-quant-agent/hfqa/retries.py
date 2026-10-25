"""Bounded durable retries for transport faults; validation and cost gates never change."""
from datetime import datetime,timezone
import errno,http.client,json,math,re,socket,time,urllib.error
from email.utils import parsedate_to_datetime
from hfqa.jobs import TRANSITIONS,TransitionError,utcnow

DELAYS=(30,120,600)
RETRYABLE_STATES=frozenset({'CREATED','INSPECTING','PLANNING','WAITING_FOR_WORKER','QUANTIZING','VALIDATING','UPLOADING','VERIFYING'})
HTTP_RETRY=frozenset({408,425,429,500,502,503,504})
NETWORK_ERRNOS={errno.ECONNRESET,errno.ECONNABORTED,errno.ETIMEDOUT,errno.EPIPE,errno.ENETUNREACH,errno.EHOSTUNREACH,errno.ECONNREFUSED}

class TransientTransferError(OSError):pass

def _save_state(store,job,target,retry,error,message,data=None):
    if target not in TRANSITIONS[job['state']]:raise TransitionError('invalid automatic retry transition')
    store.conn.execute('UPDATE jobs SET state=?,retry_json=?,error=?,paused_from=NULL,updated_at=? WHERE id=?',
                       (target,json.dumps(retry),error,utcnow(),job['id']))
    store._event(job['id'],job['state'],target,message,data)

def classify(error):
    """Inspect typed cause chains, not arbitrary human error prose."""
    current=error;seen=set()
    while isinstance(current,BaseException) and id(current) not in seen and len(seen)<8:
        seen.add(id(current))
        response=getattr(current,'response',None)
        status=getattr(current,'code',None) if isinstance(current,urllib.error.HTTPError) else getattr(response,'status_code',None)
        if type(status) is int:
            if status not in HTTP_RETRY:return None
            headers=getattr(current,'headers',None) or getattr(response,'headers',None) or {}
            after=headers.get('Retry-After')
            delay=0
            if after:
                try:delay=float(after)
                except (TypeError,ValueError):
                    try:delay=parsedate_to_datetime(after).timestamp()-time.time()
                    except (TypeError,ValueError,OverflowError):pass
            if not math.isfinite(delay):delay=0
            return {'code':'http_'+str(status),'retry_after_seconds':max(0,min(delay,86400))}
        module=type(current).__module__;name=type(current).__name__
        if (isinstance(current,(TimeoutError,ConnectionError,TransientTransferError,http.client.IncompleteRead))
            or isinstance(current,socket.gaierror) and current.errno==socket.EAI_AGAIN
            or isinstance(current,OSError) and current.errno in NETWORK_ERRNOS
            or module.startswith(('httpx','httpcore','requests','urllib3')) and name in {'ConnectError','ReadError','WriteError','NetworkError','ConnectTimeout','ReadTimeout','WriteTimeout','PoolTimeout','ConnectionError','Timeout','ReadTimeoutError','ProtocolError','RemoteProtocolError'}):
            return {'code':'temporary_network_error','retry_after_seconds':0}
        # hf-xet exposes some Rust transport failures only as RuntimeError.
        if isinstance(current,RuntimeError) and str(current).startswith('Data processing error:') and 'CAS service error' in str(current):
            match=re.search(r'HTTP status (?:server |client )?error \((\d{3})',str(current))
            if match:return {'code':'http_'+match[1],'retry_after_seconds':0} if int(match[1]) in HTTP_RETRY else None
            if re.search(r'TimedOut|ConnectionReset|connect error|Request failed after \d+ retries|operation timed out|error sending request',str(current),re.I):
                return {'code':'temporary_network_error','retry_after_seconds':0}
        reason=getattr(current,'reason',None) if isinstance(current,urllib.error.URLError) else None
        current=current.__cause__ or (reason if isinstance(reason,BaseException) else None)
    return None

def schedule(store,job_id,error,*,now=None):
    fault=classify(error)
    if fault is None:return False
    now=time.time() if now is None else now
    with store._write():
        job=store.get(job_id)
        if not job:return False
        if job['state']=='RETRY_WAIT':return True
        if job['state'] not in RETRYABLE_STATES:return False
        retry=dict(job.get('retry') or {});attempts=int(retry.get('attempts') or 0)
        # URLs can carry signed credentials. Only a classified code is exposed.
        retry.update(error_code=fault['code'],last_failure_at=datetime.fromtimestamp(now,timezone.utc).isoformat(),max_retries=len(DELAYS),resume_state=job['state'])
        if attempts>=len(DELAYS):
            retry.update(status='exhausted',next_attempt_at=None,next_attempt_epoch=None)
            _save_state(store,job,'BLOCKED',retry,'Automatische Wiederholungen ausgeschöpft; letzter Fehler: '+fault['code'],
                        'automatic transport retry budget exhausted; checkpoints retained',{'error_code':fault['code'],'attempts':attempts})
            return True
        delay=max(DELAYS[attempts],fault['retry_after_seconds']);deadline=now+delay
        retry.update(attempts=attempts+1,status='waiting',next_attempt_epoch=deadline,next_attempt_at=datetime.fromtimestamp(deadline,timezone.utc).isoformat())
        _save_state(store,job,'RETRY_WAIT',retry,None,'temporary transport failure; automatic retry scheduled',{'error_code':fault['code'],'attempt':attempts+1,'next_attempt_at':retry['next_attempt_at'],'resume_state':job['state']})
        return True

def resume_due(store,*,now=None):
    now=time.time() if now is None else now;resumed=[]
    for job in sorted(store.list_jobs('RETRY_WAIT'),key=lambda j:j['queue_position']):
        with store._write():
            job=store.get(job['id']);retry=dict(job.get('retry') or {})
            if job['state']!='RETRY_WAIT' or retry.get('status')!='waiting' or float(retry.get('next_attempt_epoch') or float('inf'))>now:continue
            if store.conn.execute('SELECT 1 FROM job_leases WHERE job_id=? AND expires_at>?',(job['id'],now)).fetchone():continue
            if store.conn.execute('SELECT 1 FROM execution_leases WHERE job_id=? AND expires_at>?',(job['id'],now)).fetchone():continue
            state=retry.get('resume_state')
            if state not in RETRYABLE_STATES:
                _save_state(store,job,'BLOCKED',retry,'Ungültiger Wiederaufnahme-Stand','invalid automatic retry checkpoint; operator attention required');continue
            retry.update(status='retrying',next_attempt_epoch=None,next_attempt_at=None)
            _save_state(store,job,state,retry,None,'automatic retry resumes exact phase; artifacts and source pins retained',{'attempt':retry['attempts']})
            resumed.append(job['id'])
    return resumed
