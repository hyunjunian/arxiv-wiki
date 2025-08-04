The paper introduces EXAONE 4.0, a unified large language model series by LG AI Research that integrates two operational modes: NON-REASONING and REASONING. This integration combines the user-friendly instruction-following abilities from EXAONE 3.5 with the advanced reasoning capabilities of EXAONE Deep into a single model, enhancing both usability and performance.

Key innovations include:
- A hybrid attention mechanism that combines local (sliding window) and global attention to efficiently handle very long context lengths (up to 128K tokens for the larger model).
- Support for agentic tool use, enabling the model to integrate and interact with external tools, which is critical for future agentic AI applications.
- Multilingual support expanded to include Spanish alongside English and Korean, without sacrificing performance in previously supported languages.
- Two model sizes: a 32-billion parameter version optimized for high performance and a smaller 1.2-billion parameter version suitable for on-device applications.

The training process features extensive supervised fine-tuning across multiple domains (world knowledge, math/code/logic, tool use, long context, and multilinguality) supplemented by reinforcement learning to boost reasoning performance, notably with an advanced algorithm called AGAPO. Preference learning is also applied to better align reasoning and non-reasoning modes.

EXAONE 4.0 demonstrates superior or competitive performance compared to other open-weight models and some frontier large models across a variety of benchmarks involving knowledge, reasoning, long-context understanding, instruction following, multilingual tasks, and agentic tool use.

In sum, EXAONE 4.0 advances large language model capabilities by unifying reasoning and non-reasoning modes, incorporating tool use, extending language support, and enabling effective handling of extremely long contexts in a publicly available research model suite.