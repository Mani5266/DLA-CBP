# E1-E12 + variants runner (reduced budget: 15% data / 8 epochs each).
# Each run appends to outputs/results.json and writes history to outputs/logs/.
# Reused (not re-run): E3_d03 as E4-off/E6-256/E7-64/E11-mid; Phase 7 for E8;
# Phase 5/vars for E9; Phases 4-6 for E10. Runtimes ~30-60s each on RTX 3050.
$ErrorActionPreference = "Stop"
function Run-Exp($args) { python src/train.py @args }

Run-Exp @("--config","exp","--tag","E12_gpu","--subset","0.05","--epochs","2","--batch","64","--stride","2")
Run-Exp @("--config","exp","--tag","var_many2many","--model","many2many")
Run-Exp @("--config","exp","--tag","var_stateful","--model","stateful")
Run-Exp @("--config","exp","--tag","var_cnn","--model","cnn")
Run-Exp @("--config","exp","--tag","E1_sgd","--opt","sgd")
Run-Exp @("--config","exp","--tag","E1_sgd_mom","--opt","sgd_mom")
Run-Exp @("--config","exp","--tag","E1_nag","--opt","nag")
Run-Exp @("--config","exp","--tag","E1_adagrad","--opt","adagrad")
Run-Exp @("--config","exp","--tag","E1_adadelta","--opt","adadelta")
Run-Exp @("--config","exp","--tag","E1_rmsprop","--opt","rmsprop")
Run-Exp @("--config","exp","--tag","E1_adam","--opt","adam")
Run-Exp @("--config","exp","--tag","E2_clip_on","--lr","0.01","--clip","1.0")
Run-Exp @("--config","exp","--tag","E2_clip_off","--lr","0.01","--clip","0")
Run-Exp @("--config","exp","--tag","E3_d0","--dropout","0")
Run-Exp @("--config","exp","--tag","E3_d03","--dropout","0.3")
Run-Exp @("--config","exp","--tag","E3_d05","--dropout","0.5")
Run-Exp @("--config","exp","--tag","E4_bn","--batchnorm")
Run-Exp @("--config","exp","--tag","E6_u128","--units","128")
Run-Exp @("--config","exp","--tag","E6_u512","--units","512")
Run-Exp @("--config","exp","--tag","E7_s32","--seq-len","32")
Run-Exp @("--config","exp","--tag","E7_s100","--seq-len","100")
Run-Exp @("--config","exp","--tag","E11_b32_lr1e-3","--batch","32","--lr","0.001")
Run-Exp @("--config","exp","--tag","E11_b64_lr1e-3","--batch","64","--lr","0.001")
Run-Exp @("--config","exp","--tag","E11_b32_lr3e-3","--batch","32","--lr","0.003")
Run-Exp @("--config","exp","--tag","E11_b32_lr1e-2","--batch","32","--lr","0.01")
Run-Exp @("--config","exp","--tag","E11_b64_lr1e-2","--batch","64","--lr","0.01")
Write-Output "ALL_EXPERIMENTS_DONE"
