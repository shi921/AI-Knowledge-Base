"""Local prompt templates; API use only when explicit --api-base given."""
import argparse, json
from urllib.parse import urljoin
from urllib.request import Request, urlopen

def prompts(text):
    return {
        'zero_shot': 'Classify the sentiment as P or N. Answer with one letter only. Review: '+text,
        'few_shot': ('Classify movie reviews. Answer only P or N.\n'
            'Review: a beautiful moving movie\nSentiment: P\n'
            'Review: an awful boring film\nSentiment: N\n'
            f'Review: {text}\nSentiment:'),
        'reasoning_check': ('Solve the task carefully and verify the answer before finalizing. '
            'Give a concise final answer.\nTask: '+text),
    }

def main():
    ap=argparse.ArgumentParser();ap.add_argument('text',nargs='?',default='A very boring and disappointing movie.')
    ap.add_argument('--api-base',help='Opt in to external text generation; URL ending in /api/')
    ap.add_argument('--model',default='qwen3:4b');args=ap.parse_args()
    for mode,pt in prompts(args.text).items():
        print('\n###',mode,'\n',pt)
        if args.api_base:
            req=Request(urljoin(args.api_base.rstrip('/')+'/', 'generate'),
                data=json.dumps({'model':args.model,'prompt':pt,'stream':False}).encode('utf-8'),
                headers={'Content-Type':'application/json'},method='POST')
            try:
                with urlopen(req,timeout=45) as r:body=json.load(r)
                print('ANSWER:',body.get('response',body))
            except Exception as ex:print('external API error:',ex)
if __name__=='__main__':main()
