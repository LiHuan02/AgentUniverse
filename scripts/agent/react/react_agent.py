import os
import re
import json
from dotenv import load_dotenv
from scripts.base.my_llm import MyLLM
from scripts.agent.react.toolbox import ToolExecutor

load_dotenv()

REACT_PROMPT_TEMPLATE = """
你是一个有能力调用外部工具的智能助手。

可用工具如下:
{tools}

请严格按照以下格式进行回应，每次只能输出一组 Thought 和 Action：

Thought: 你的思考过程，分析当前情况，决定下一步。
Action: 你决定调用的工具，格式如下之一：
- 无参数调用: tool_name[]
- 单参数调用（不要加引号！）: tool_name[参数值]
- 多参数调用（必须用JSON）: tool_name[{{"参数名1": "参数值1"}}]
- 最终答案: Finish[最终答案]

⚠️ 严格规则 ⚠️
1. 必须且只能以 "Thought:" 开头！
2. 每次只能输出一个 Thought 和一个 Action！
4. 输出Finish后立即结束对话，不要再输出任何内容。
4. Action 必须单独占一行。
5. 单参数调用时，参数值绝对不要加双引号或单引号！例如写 Search[A股行情]，不要写 Search["A股行情"]！

现在，请开始解决以下问题:
Question: {question}
History: {history}
"""



class ReActAgent:
    def __init__(self, llm: MyLLM, tools: ToolExecutor, max_turns: int = 5):
        """
        初始化ReAct智能体。
        llm: 用于思考的语言模型客户端。
        tools: 可用工具的执行器。
        max_turns: 智能体思考的最大轮数，防止无限循环。
        """
        self.llm = llm
        self.tools = tools
        self.max_turns = max_turns
        self.history = []

    def run(self, user_input: str) -> str:
        """
        运行ReAct智能体回答一个问题
        """
        # 重置思考历史
        self.history = []
        current_turn = 0
        while current_turn < self.max_turns:
            current_turn += 1
            print(f"\n=== 第 {current_turn} 轮思考 ===")

            # 构建prompt
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=self.tools.getAvailableTools(),
                question=user_input,
                history="\n".join(self.history)
            )

            # 调用LLM进行思考
            llm_output = self.llm.think([{"role": "system", "content": prompt}])
            if not llm_output:
                return "抱歉，智能体无法生成回应。"
            
            # 解析LLM输出
            _, action = self._parse_output(llm_output)
            if not action:
                return "抱歉，智能体Action异常，无法继续。"  

            # 处理Action
            if action.startswith("Finish[") and action.endswith("]"):
                final_answer = action[len("Finish["):-1].strip()
                print(f"最终答案: {final_answer}")
                return final_answer

            tool_name, tool_input = self._parse_action(action)
            if tool_name and tool_input is not None:
                tool_func = self.tools.getTool(tool_name)
                if tool_func:
                    print(f"调用工具: {tool_name} 输入: {tool_input}")
                    
                    # 根据参数类型智能调用
                    if isinstance(tool_input, dict):
                        # 如果是多参数JSON，解包为关键字参数，如 search(query="A股", engine="baidu")
                        tool_result = tool_func(**tool_input)
                    elif tool_input == "":  # 新增：如果是空字符串，说明无参数
                        tool_result = tool_func()
                    else:
                        # 如果是单参数字符串，直接作为位置参数传入
                        tool_result = tool_func(tool_input)
                        
                    print(f"工具结果: {tool_result}")
                    self.history.append(f"工具调用: {tool_name}[{tool_input}] -> {tool_result}")
                else:
                    print(f"未知工具: {tool_name}，请检查工具名称是否正确。")
                    self.history.append(f"未知工具: {tool_name}，无法执行。")
            else:
                print("无法解析Action中的工具调用，请确保格式正确。")
                self.history.append("无法解析Action中的工具调用，跳过执行。")  
            
        return "抱歉，智能体未能在规定的轮数内得出结论。"


    def _parse_output(self, text: str):
        """
        解析LLM的输出，提取第一组 Thought 和 Action。
        """
        # Thought: 匹配到第一个 Action: 为止
        thought_match = re.search(r"Thought:\s*(.*?)(?=\n\s*Action:)", text, re.DOTALL)
        # Action: 只取这一行的内容（不加 DOTALL，. 不匹配换行符，确保只取一行）
        action_match = re.search(r"Action:\s*(.*)", text)

        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_action(self, action_text: str):
        """
        解析Action字符串，提取工具名称和输入。
        注意：不要用 re.DOTALL，避免跨行匹配吞入垃圾内容。
        """
        # 不加 DOTALL，. 不匹配换行，确保 [] 内只取当前行内容
        match = re.match(r"(\w+)\[(.*)\]", action_text)
        if match:
            tool_name = match.group(1)
            input_str = match.group(2).strip()
            
            # 尝试解析为 JSON (处理多参数)
            try:
                tool_input = json.loads(input_str)
                return tool_name, tool_input
            except json.JSONDecodeError:
                return tool_name, input_str
                
        return None, None

