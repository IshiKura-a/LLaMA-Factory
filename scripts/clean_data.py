import pandas as pd
import re
import os

def main():
    mnt_dir = f'/mnt/zh_blob/dataset/ads_sft'
    input_file = os.path.join(mnt_dir, f'eem_ds_test.jsonl')
    output_file = os.path.join(mnt_dir, f'eem_ds_test_cleaned.jsonl')
    
    df = pd.read_json(input_file, lines=True, orient='records')
    pattern = re.compile(
        r'<\uff5cbegin\u2581of\u2581sentence\uff5c>(.*?)<\uff5cUser\uff5c>(.*)',
        re.UNICODE | re.DOTALL  # Allows matching across multiple lines
    )

    # Extract system and user text
    df[['system', 'prompt']] = df['prompt'].str.extract(pattern)
    df['system'] = df['system'].apply(lambda x: x.strip().replace('\u00ae', ''))
    df['prompt'] = df['prompt'].apply(lambda x: x.strip().replace('\u00ae', ''))
    df['response'] = df['completion'].apply(lambda x: x.strip().replace('<\uff5cAssistant\uff5c>', '').replace('<\uff5cend\u2581of\u2581sentence\uff5c>', ''))
    del df['completion']

    df.to_json(output_file, lines=True, orient='records')

if __name__ == '__main__':
    main()
