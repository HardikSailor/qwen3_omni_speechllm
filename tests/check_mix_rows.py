"""Build one row from 2 samples of every dataset in a mix (train + validation) with the run's prompt settings; reports failures
and the language each dataset resolves to. CPU only, inside the container:
    $RUN python tests/check_mix_rows.py mixes/mv4_v0.yaml"""
import json, os, struct, sys, random, collections
sys.path.insert(0, '.')
import zstandard
from omni_mds.mix import load_mix
from omni_mds.sea_text import RowConfig, build_row, resolve_languages
def first_sample(p):
    sh=json.load(open(os.path.join(p,'index.json')))['shards'][0]
    zf=os.path.join(p, (sh.get('zip_data') or {}).get('basename','#'))
    raw = zstandard.ZstdDecompressor().decompress(open(zf,'rb').read(), max_output_size=1<<30) if os.path.exists(zf) else open(os.path.join(p, sh['raw_data']['basename']),'rb').read()
    n=struct.unpack_from('<I',raw,0)[0]; offs=struct.unpack_from(f'<{n+1}I',raw,4)
    cols, enc, sizes = sh['column_names'], sh['column_encodings'], sh['column_sizes']
    out=[]
    for i in (0, n//2):
        b=raw[offs[i]:offs[i+1]]; nvar=sum(s is None for s in sizes); var=struct.unpack_from(f'<{nvar}I',b,0); q=4*nvar; vi=0; t={}
        for c,e,s in zip(cols,enc,sizes):
            z = var[vi] if s is None else s; vi += s is None
            v=b[q:q+z]; q+=z; t[c]= v if e=='bytes' else v.decode()
        out.append(t)
    return out
cfg=RowConfig(asr_prompt_type='random_multilingual', ac_prompt_type='random_multilingual', augment_prompt=True)
m=load_mix(sys.argv[1]); bad=0; langs=collections.Counter()
for split in ('train','validation'):
    for s in m[split]:
        try:
            for smp in first_sample(s.path):
                assert smp.get('context_audio'), 'no context_audio'
                row=build_row(smp, s.meta(), cfg, random.Random(0), is_train=(split=='train'), with_meta=True)
                lang=row['_meta'].get('language'); langs[(s.task, lang)]+=1
        except Exception as e:
            bad+=1; print('FAIL', split, s.name, repr(e)[:200])
print('failures', bad); print(sorted(langs.items(), key=lambda x:-x[1])[:60])
