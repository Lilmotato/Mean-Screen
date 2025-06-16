import asyncio
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.services.llm_services import DIALService

async def main():
    try:
        dial = DIALService()
        text = "I hate you and everything you stand for."
        result = await dial.classify_text(text)
        print("🧪 Classification Result:")
        print(result)
   
    except Exception as e:
        print(f"❗ Unexpected error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
