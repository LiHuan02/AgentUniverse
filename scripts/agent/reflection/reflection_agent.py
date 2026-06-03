from scripts.base.my_llm import MyLLM
from memory import Memory


# 初始提示词模板
INITIAL_PROMPT_TEMPLATE = """
你是一位资深的Python程序员。请根据以下要求，编写一个Python函数。
你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。

要求: {task}

请直接输出代码，不要包含任何额外的解释。
"""


# 反思提示词模板
REFLECT_PROMPT_TEMPLATE = """
你是一位严格的代码评审专家和资深算法工程师，对代码的性能与简洁有极致的要求。
你的任务是审查以下Python代码，并专注于找出代码的可能问题或优化方向。

# 原始任务:
{task}

# 待审查的代码:
```python
{code}
```

请清晰地指出当前算法的不足，并提出具体的、可行的改进算法建议。
如果代码在算法层面已经达到最优，则回答“无需改进”。

请直接输出你的反馈，不要包含任何额外的解释。
"""


# 优化提示词模板

REFINE_PROMPT_TEMPLATE = """
你是一位资深的Python程序员。你正在根据一位代码评审专家的反馈来优化你的代码。
你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。

# 原始任务:
{task}

# 你上一轮尝试的代码:
{last_code_attempt}
评审员的反馈：
{feedback}

请根据评审员的反馈，生成一个优化后的新版本代码。
请直接输出优化后的代码，不要包含任何额外的解释。
"""

class ReflectionAgent:
    def __init__(self, llm, max_iter=3):
        self.llm = llm
        self.memory = Memory()
        self.max_iter = max_iter

    def _get_llm_response(self, prompt: str) -> str:
        """一个辅助方法，用于调用LLM并获取完整的流式响应。"""
        messages = [{"role": "user", "content": prompt}]
        response_text = self.llm.think(messages=messages, stream=True) or ""
        return response_text

    def run(self, task: str):
        print(f"\n--- 开始处理任务 ---\n任务: {task}")
        success = False

        # --- 1. 初始执行 ---
        print("\n--- 正在进行初始尝试 ---")
        initial_prompt = INITIAL_PROMPT_TEMPLATE.format(task=task)
        initial_code = self._get_llm_response(initial_prompt)
        self.memory.add_record("execution", initial_code)

        # --- 2. 迭代循环:反思与优化 ---
        for i in range(self.max_iter):
            print(f"\n--- 第 {i+1}/{self.max_iter} 轮迭代 ---")

            # a. 反思
            print("\n-> 正在进行反思...")
            last_code = self.memory.get_last_execution()
            reflect_prompt = REFLECT_PROMPT_TEMPLATE.format(task=task, code=last_code)
            feedback = self._get_llm_response(reflect_prompt)
            self.memory.add_record("reflection", feedback)

            # b. 检查是否需要停止
            if "无需改进" in feedback:
                print("\n✅ 反思认为代码已无需改进，任务完成。")
                success = True
                break

            # c. 优化
            print("\n-> 正在进行优化...")
            refine_prompt = REFINE_PROMPT_TEMPLATE.format(
                task=task,
                last_code_attempt=last_code,
                feedback=feedback
            )
            refined_code = self._get_llm_response(refine_prompt)
            self.memory.add_record("execution", refined_code)
        
        final_code = self.memory.get_last_execution()
        if success:
            print(f"\n--- 任务完成 ---\n最终生成的代码:\n```python\n{final_code}\n```")
        else:
            print(f"\n--- 未能在有限次数达到陪审员要求 ---\n最终生成的代码:\n```python\n{final_code}\n```")
        return final_code