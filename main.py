import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class MessageDataset(Dataset):
    """메시지 데이터셋 클래스"""
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # 데이터셋의 각 행
        row = self.data.iloc[index]
        premise = f"'{row['relationship']}' 관계에서 메시지가 적합한가요?"
        hypothesis = row['message']

        # 토큰화
        inputs = self.tokenizer(
            premise,
            hypothesis,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )

        # 반환 데이터 (input_ids, attention_mask, label)
        return {
            'input_ids': inputs['input_ids'].squeeze(0),  # (1, MAX_LEN) → (MAX_LEN)
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': torch.tensor(row['label'], dtype=torch.long)
        }

def load_data(file_path):
    """
    CSV 파일에서 데이터 로드
    파일 형식: message, relationship, label
    label: 0 (부적절), 1 (중립), 2 (적절)
    """
    try:
        df = pd.read_csv(file_path)
        print(f"데이터 로드 완료: {len(df)}개 샘플")
        assert all(col in df.columns for col in ['message', 'relationship', 'label'])
        return df
    except Exception as e:
        print("데이터 로드 에러: CSV 파일 형식이 잘못됐거나 파일 경로를 확인하세요.")
        raise e

def train_model(train_file, model_name='snunlp/KR-FinBERT', batch_size=16, epochs=3, output_dir="./model_output"):
    """
    모델 학습
    train_file: 학습 데이터가 저장된 CSV 파일 경로
    """
    # 1. 데이터 로드
    df = load_data(train_file)

    # 2. 모델 및 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    # 3. 데이터셋 생성
    dataset = MessageDataset(df, tokenizer)

    # 4. 학습 설정
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",  # 매 에포크마다 평가
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_total_limit=2,
        save_steps=200,
        report_to="none",  # WandB, Tensorboard 사용 안 함
        load_best_model_at_end=True
    )

    # 5. 트레이너 설정 및 학습
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer
    )

    print("학습 시작...")
    trainer.train()
    print("학습 완료!")

    # 학습 완료된 모델 저장
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"모델과 토크나이저 저장 완료: {output_dir}")

def evaluate_model(test_file, model_dir, model_name='snunlp/KR-FinBERT'):
    """
    학습된 모델로 테스트 데이터 평가
    """
    # 1. 데이터 로드
    df = load_data(test_file)

    # 2. 모델 및 토크나이저 로드
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # 3. 평가
    model.eval()
    correct = 0
    total = len(df)
    for _, row in df.iterrows():
        premise = f"'{row['relationship']}' 관계에서 메시지가 적합한가요?"
        hypothesis = row['message']

        encoded = tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            max_length=128,
            truncation=True,
            padding='max_length'
        )
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']

        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            predicted = torch.argmax(outputs.logits, dim=-1).item()

        if predicted == row['label']:
            correct += 1

    accuracy = correct / total * 100
    print(f"정확도: {accuracy:.2f}%")
    return accuracy

if __name__ == "__main__":
    # 학습 및 평가 실행 (파일 경로를 지정하세요)
    train_file = "train_data.csv"  # 학습 데이터 CSV 파일 경로
    test_file = "test_data.csv"    # 테스트 데이터 CSV 파일 경로
    model_dir = "./model_output"  # 학습된 모델이 저장될 디렉토리

    # 1. 모델 학습
    train_model(train_file)

    # 2. 모델 평가
    evaluate_model(test_file, model_dir)
