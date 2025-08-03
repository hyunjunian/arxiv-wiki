The paper "Beyond Memorization: Violating Privacy via Inference with Large Language Models," presented at ICLR 2024 by researchers from ETH Zurich, investigates privacy risks posed by large language models (LLMs) beyond just memorization of training data.

Key content and findings:

1. **New Privacy Threat - Inference:** The paper studies how pretrained LLMs can infer personal attributes (e.g., location, age, sex, income) from user-generated text given at inference time, without relying on memorized training data.

2. **PersonalReddit Dataset:** The authors curated a dataset of real Reddit user profiles with texts and manually labeled personal attributes across eight categories such as location, income, education, and relationship status.

3. **Strong Prediction Performance:** State-of-the-art LLMs including GPT-4 achieve about 85% top-1 and 95% top-3 accuracy on inferring personal attributes, close to or surpassing human labelers, but at a fraction of the cost and time (100× less money and 240× less time).

4. **Emerging Threat of Malicious Chatbots:** The paper formalizes and simulates how adversarial chatbots can steer conversations to extract sensitive user information through seemingly benign dialogue, showing a top-1 accuracy around 60% for inferred attributes.

5. **Ineffectiveness of Current Mitigations:**
   - State-of-the-art text anonymization tools do not sufficiently protect privacy as LLMs can infer attributes from subtle language cues left in anonymized text.
   - Provider-side alignment of models (e.g., safety filtering) does not adequately prevent privacy-invasive prompts.

6. **Implications:** This exposes a scalable new privacy risk where adversaries can extract personal data at scale using off-the-shelf LLMs, not limited to memorized data extraction.

7. **Call for Action:** The authors advocate for developing improved privacy protections beyond memorization defenses, including stronger anonymization methods, better privacy-aligned LLMs, and more research focus on inference-based privacy risks.

8. **Responsible Disclosure and Resource Release:** The paper includes responsibly handling sensitive data, synthetic datasets for research, and code availability to encourage further study.

In summary, the paper reveals that current pretrained large language models have powerful inference capabilities that enable privacy-violating extraction of personal information from users' text, posing serious privacy risks not tackled by existing defenses and calling for a broader discussion and technical advancements in privacy protection.