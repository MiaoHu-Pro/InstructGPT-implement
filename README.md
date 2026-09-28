# InstructGPT implementation

This directory is a compact, educational implementation of the main
InstructGPT/RLHF training workflow. InstructGPT improves a pretrained language
model by first teaching it the desired text distribution, then learning a
reward signal, and finally optimizing the language model against that reward
while limiting how far it moves from the supervised model.

The three main stages are:

1. **Supervised fine-tuning (SFT)** — `1-SFT-fixed.py` fine-tunes a Chinese
   GPT-2 model with next-token prediction and saves the SFT policy.
2. **Reward-model training (RM)** — `2-RM-fixed.py` initializes from the SFT
   model, adds a scalar reward head, and learns to score generated text.
3. **RLHF with PPO** — `3-PPO-fixed.py` generates responses with the SFT
   policy, scores them with the frozen reward model, and updates an
   actor-critic model using PPO, GAE, value loss, clipped policy ratios, and a
   KL penalty against a frozen reference policy.

The intended execution order is therefore:

```text
pretrained GPT-2 -> SFT policy -> reward model -> PPO/RLHF policy
```

The `*-fixed.py` files are the recommended versions, and the corresponding
`submit-*-fixed.sh` files run them on the server. `2-DPO.py` demonstrates DPO
as a separate preference-optimization method; DPO is not part of the original
InstructGPT PPO pipeline.

This project demonstrates the algorithmic structure rather than reproducing
the full production InstructGPT system. In particular, SFT uses Chinese review
language modeling rather than instruction-response examples, and the reward
model uses binary sentiment labels rather than human-ranked response pairs.
