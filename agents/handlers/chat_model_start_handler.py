from langchain.callbacks.base import BaseCallbackHandler
from pyboxen import boxen


def boxen_print(*args, **kwargs):
    print(boxen(*args, **kwargs))


class ChatModelStartHandler(BaseCallbackHandler):
    def on_chat_model_start(self, serialized, messages, **kwargs):
        print("\n\n\n\n========= Sending Messages =========\n\n")

        for message in messages[0]:
            if message.type == "system":
                boxen_print(message.content, title=message.type, color="yellow")

            elif message.type == "human":
                boxen_print(message.content, title=message.type, color="green")
            
            elif message.type in ["ai", "AIMessageChunk"] and "tool_calls" in message.additional_kwargs:
                for tool_call in message.additional_kwargs["tool_calls"]:
                    boxen_print(
                        f"Running tool {tool_call['function']['name']} with args {tool_call['function']['arguments']}",
                        title="ai",
                        color="cyan"
                    )

            elif message.type == "ai":
                boxen_print(message.content, title=message.type, color="blue")
            
            elif message.type == "tool":
                boxen_print(message.content, title=message.type, color="purple")

            else:
                # For other message types, check if there's content to display
                if hasattr(message, 'content') and message.content:
                    boxen_print(message.content, title=message.type)