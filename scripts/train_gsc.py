#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, random, sys, time
from pathlib import Path
REPO_ROOT=Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0,str(REPO_ROOT))
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from reptor import reptor9, reptor_a
from reptor.data import GSCDataset, LABELS


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def build_model(name):
    return reptor9(num_classes=len(LABELS)) if name=="reptor9" else reptor_a(num_classes=len(LABELS))


def lr_for_epoch(epoch,total,base,warmup):
    if warmup>0 and epoch<warmup: return base*(epoch+1)/warmup
    span=max(1,total-warmup); p=min(max((epoch-warmup)/span,0.0),1.0)
    return 0.5*base*(1+math.cos(math.pi*p))


def set_lr(opt,lr):
    for g in opt.param_groups: g["lr"]=lr

@torch.no_grad()
def evaluate(model,loader,criterion,device):
    model.eval(); loss_sum=correct=count=0
    for x,y in loader:
        x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
        z=model(x); loss=criterion(z,y); b=y.size(0)
        loss_sum+=loss.item()*b; correct+=(z.argmax(1)==y).sum().item(); count+=b
    return loss_sum/count, correct/count


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-root",type=Path,required=True)
    ap.add_argument("--manifest-dir",type=Path,default=Path("manifests"))
    ap.add_argument("--model",choices=["reptor9","reptor_a"],default="reptor9")
    ap.add_argument("--epochs",type=int,default=1)
    ap.add_argument("--batch-size",type=int,default=128)
    ap.add_argument("--lr",type=float,default=0.02)
    ap.add_argument("--momentum",type=float,default=0.9)
    ap.add_argument("--weight-decay",type=float,default=1e-4)
    ap.add_argument("--warmup-epochs",type=int,default=5,
                    help="Paper says 5 warm-up steps; this script interprets them as epochs (explicit reproduction choice).")
    ap.add_argument("--num-workers",type=int,default=2)
    ap.add_argument("--seed",type=int,default=1337)
    ap.add_argument("--output-dir",type=Path,default=Path("runs"))
    ap.add_argument("--time-shift-ms",type=float,default=100.0)
    ap.add_argument("--noise-prob",type=float,default=0.8)
    ap.add_argument("--noise-max-scale",type=float,default=0.1)
    ap.add_argument("--freq-mask-param",type=int,default=6)
    ap.add_argument("--time-mask-param",type=int,default=10)
    ap.add_argument("--no-specaugment",action="store_true")
    ap.add_argument("--test-at-end",action="store_true")
    args=ap.parse_args(); seed_all(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin=device.type=="cuda"
    run=args.output_dir/f"{args.model}_seed{args.seed}_ep{args.epochs}"; run.mkdir(parents=True,exist_ok=True)

    common=dict(data_root=args.data_root)
    train=GSCDataset(**common,manifest_path=args.manifest_dir/"gsc12_train.csv",training=True,
                     time_shift_ms=args.time_shift_ms,noise_prob=args.noise_prob,noise_max_scale=args.noise_max_scale,
                     specaugment=not args.no_specaugment,freq_mask_param=args.freq_mask_param,time_mask_param=args.time_mask_param)
    val=GSCDataset(**common,manifest_path=args.manifest_dir/"gsc12_val.csv",training=False,specaugment=False)
    test=GSCDataset(**common,manifest_path=args.manifest_dir/"gsc12_test.csv",training=False,specaugment=False)
    def loader(ds,shuffle):
        return DataLoader(ds,batch_size=args.batch_size,shuffle=shuffle,num_workers=args.num_workers,
                          pin_memory=pin,persistent_workers=args.num_workers>0)
    tr,va,te=loader(train,True),loader(val,False),loader(test,False)
    model=build_model(args.model).to(device); crit=nn.CrossEntropyLoss()
    opt=torch.optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
    best=-1.0; hist=[]; total0=time.perf_counter()
    print(f"device={device} model={args.model} train/val/test={len(train)}/{len(val)}/{len(test)}")

    for ep in range(args.epochs):
        lr=lr_for_epoch(ep,args.epochs,args.lr,min(args.warmup_epochs,args.epochs)); set_lr(opt,lr)
        model.train(); t0=time.perf_counter(); loss_sum=correct=count=0
        for x,y in tr:
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True); z=model(x); loss=crit(z,y); loss.backward(); opt.step()
            b=y.size(0); loss_sum+=loss.item()*b; correct+=(z.argmax(1)==y).sum().item(); count+=b
        tl,ta=loss_sum/count,correct/count; vl,va_acc=evaluate(model,va,crit,device); sec=time.perf_counter()-t0
        rec={"epoch":ep+1,"lr":lr,"train_loss":tl,"train_acc":ta,"val_loss":vl,"val_acc":va_acc,"seconds":sec}; hist.append(rec)
        print(f"epoch {ep+1:03d}/{args.epochs:03d} lr={lr:.6f} train={tl:.4f}/{ta*100:.2f}% val={vl:.4f}/{va_acc*100:.2f}% time={sec:.1f}s")
        ck={"model_name":args.model,"model_state":model.state_dict(),"optimizer_state":opt.state_dict(),
            "epoch":ep+1,"val_acc":va_acc,"labels":LABELS,"args":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}}
        torch.save(ck,run/"last.pt")
        if va_acc>best: best=va_acc; torch.save(ck,run/"best.pt")
        (run/"history.json").write_text(json.dumps(hist,indent=2),encoding="utf-8")

    print(f"best val={best*100:.2f}% total={(time.perf_counter()-total0)/60:.2f} min run={run}")
    if args.test_at_end:
        ck=torch.load(run/"best.pt",map_location=device,weights_only=False); model.load_state_dict(ck["model_state"])
        tl,ta=evaluate(model,te,crit,device); print(f"TEST loss={tl:.4f} accuracy={ta*100:.2f}%")
        (run/"test_result.json").write_text(json.dumps({"test_loss":tl,"test_acc":ta,"best_val_acc":best},indent=2),encoding="utf-8")

if __name__=="__main__": main()
