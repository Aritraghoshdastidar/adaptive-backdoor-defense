"""
notebooks/colab_sk_aggressive_nad_patch.py

Colab Code Patch: Optional Overnight Exploration on Silent Killer pr01.
Paste this code cell directly into `sk_defense_v4.ipynb` immediately after Cell D3.

Purpose:
Tests whether a more aggressive fine-tuning schedule can break the SK backdoor in
the NAD teacher (which previously retained 60% ASR at lr=0.01, ep=10).
"""

# ==============================================================================
# Cell D3-Patch — Aggressive Teacher Exploration on Silent Killer (pr01 Smoke Test)
# ==============================================================================

if IN_COLAB and 'pr01' in TAGS:
    print("=" * 75)
    print("EXPLORATORY: Aggressive NAD Teacher Schedules for Silent Killer pr01")
    print("=" * 75)
    
    test_tag = 'pr01'
    base_model = load_model(sk_ckpt(test_tag))
    
    # Grid of more aggressive teacher fine-tuning recipes
    CANDIDATE_RECIPES = [
        {"name": "Aggressive-LR",  "lr": 0.05, "epochs": 20, "max_ca_drop": 0.08},
        {"name": "Long-Schedule",  "lr": 0.02, "epochs": 30, "max_ca_drop": 0.08},
        {"name": "Extended-Base",  "lr": 0.01, "epochs": 30, "max_ca_drop": 0.05},
    ]
    
    teacher_results = []
    
    for r in CANDIDATE_RECIPES:
        print(f"\nEvaluating Teacher Recipe: {r['name']} (lr={r['lr']}, epochs={r['epochs']})...")
        t0 = time.time()
        
        # Build teacher with loosened CA constraint and higher compute
        try:
            teacher_cand, t_info = build_nad_teacher(
                base_model,
                defense_loader=nad_loader,
                device=DEVICE,
                epochs=r['epochs'],
                lr=r['lr'],
                augment=True,
                amp=True,
                eval_fn=_val_eval,
                val_loader=sel_loader,
                max_ca_drop=r['max_ca_drop'],
                verbose=False,
                teacher_asr_flag_threshold=0.40,
            )
            elapsed = time.time() - t0
            t_asr = t_info.get('teacher_asr', 1.0)
            t_ca = t_info.get('teacher_val_ca', 0.0)
            gate = t_info.get('teacher_gate_passed', False)
            
            print(f"  -> Result: Teacher ASR={t_asr*100:.1f}%, CA={t_ca*100:.1f}%, Gate Passed={gate} [{elapsed:.1f}s]")
            teacher_results.append((r, teacher_cand, t_info))
            
            # If teacher ASR drops below 40%, run student distillation
            if t_asr < 0.40 and gate:
                print(f"  -> SUCCESS! Teacher ASR broke below 40%. Running Student Distillation...")
                student_cand, s_res = run_nad(
                    base_model,
                    teacher_cand,
                    defense_loader=nad_loader,
                    device=DEVICE,
                    epochs=10,
                    lr=0.01,
                    verbose=False
                )
                student_eval = evaluate(student_cand)
                print(f"  [STUDENT OUTCOME] ASR_src={student_eval['ASR_src']*100:.2f}%, CA={student_eval['CA']*100:.2f}%")
                
                # Check residual against clean reference (24.8% for NAD)
                residual = (student_eval['ASR_src'] - 0.248) * 100
                print(f"  [RESIDUAL ABOVE CLEAN NAD] +{residual:.1f}pp")
            else:
                print(f"  -> Teacher still retains backdoor (ASR={t_asr*100:.1f}%). Distillation not viable.")
                
        except Exception as e:
            print(f"  -> Error evaluating recipe {r['name']}: {e}")

    print("\n" + "=" * 75)
    print("Exploration finished. Interpret results relative to the clean baseline floor.")
    print("=" * 75)
