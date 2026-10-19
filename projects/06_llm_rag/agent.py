"""Optional LangChain tool-calling example (requires an API key and langchain-openai)."""
import os

def main():
    if not os.getenv('OPENAI_API_KEY'):raise RuntimeError('Set the OPENAI_API_KEY environment variable first')
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI
    @tool
    def count_words(text:str)->int:
        """Count whitespace-delimited words in the provided text."""
        return len(text.split())
    model=ChatOpenAI(model='gpt-4o-mini',temperature=0).bind_tools([count_words])
    reply=model.invoke('Count words in: AI is changing service systems')
    if reply.tool_calls:
        for call in reply.tool_calls:
            if call['name']=='count_words':
                print({'tool':call['name'],'result':count_words.invoke(call['args'])})
    else:print('No tool call, raw response:',reply.content)

if __name__=='__main__':main()
