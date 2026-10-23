"""Compare every output tensor type against native metadata-only quantization."""
import argparse,ctypes,json,logging,re,subprocess,sys
from pathlib import Path
from producer import setup,make_model,Quantizer,assemble_plan
from source import inventory

VARIANTS=('Q2_K','Q3_K_M','Q4_K_M','Q5_K_M')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--work',type=Path,required=True);ap.add_argument('--llama',type=Path,required=True);ap.add_argument('--build',type=Path,required=True);args=ap.parse_args()
    logging.basicConfig(level=logging.WARNING,stream=sys.stderr);setup(args.llama.resolve());import gguf
    args.work.mkdir(parents=True,exist_ok=True)
    def refuse(_):raise RuntimeError('Metadata planning must not read model payloads')
    model=make_model(args.llama,args.work,refuse);q=Quantizer(args.build);plans={}
    for variant in VARIANTS:
        plan,head=assemble_plan(model,q,variant,args.work)
        (args.work/(variant+'-plan.json')).write_text(json.dumps(plan,indent=2)+'\n')
        (args.work/(variant+'-header.gguf')).write_bytes(head)
        plans[variant]={k:v for k,v in plan.items() if k!='tensors'}
    bfplan,bfhead=assemble_plan(model,q,'BF16',args.work);sparse=args.work/'metadata-only.gguf'
    try:
        with sparse.open('wb') as f:f.write(bfhead);f.truncate(bfplan['size'])
        for variant in VARIANTS:
            log=args.work/(variant+'-native.log')
            with log.open('wb') as stream:
                subprocess.run([sys.executable,str(Path(__file__).with_name('native_plan.py')),str(args.build/'lib_minimax_quant.so'),str(sparse),str(int(getattr(gguf.LlamaFileType,'MOSTLY_'+variant)))],stdout=stream,stderr=stream,check=True)
            native={}
            for line in log.read_text().splitlines():
                match=re.search(r'\[\s*\d+/\s*\d+\]\s+(\S+)\s+-.*?type =\s+(\S+),\s*size =.*',line)
                if match:
                    target=re.search(r'\((\w+)\)\s*$',line)
                    native[match[1]]=(target[1] if target else match[2]).upper()
            plan=json.loads((args.work/(variant+'-plan.json')).read_text())
            expected={t['name']:gguf.GGMLQuantizationType(t['type']).name for t in plan['tensors']}
            if native!=expected:raise RuntimeError('Native type plan differs: '+json.dumps({n:(native.get(n),v) for n,v in expected.items() if native.get(n)!=v}))
            plans[variant]['native_full_model_type_plan_verified']=True
    finally:sparse.unlink(missing_ok=True)
    result={'source_revision':inventory()['source_revision'],'plans':plans,'full_model_weights_read':False,'inference_quality_tested':False}
    (args.work/'plan-checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
