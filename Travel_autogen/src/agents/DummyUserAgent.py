from autogen_core import BaseAgent

class DummyUserAgent(BaseAgent):
    def __init__(self):
        super().__init__("User Agent")

    async def on_message(self, message, ctx):
        # Không cần xử lý gì, chỉ để cho hệ thống biết agent này tồn tại
        pass
