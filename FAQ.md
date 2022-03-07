# FAQ

1. **How to finetune the GFPGANCleanv1-NoCE-C2 (v1.2) model**

**A:** 1) The GFPGANCleanv1-NoCE-C2 (v1.2) model uses the *clean* architecture, which is more friendly for deploying.
2) This model is not directly trained. Instead, it is converted from another *bilinear* model.
3) If you want to finetune the GFPGANCleanv1-NoCE-C2 (v1.2), you need to finetune its original *bilinear* model, and then do the conversion.
