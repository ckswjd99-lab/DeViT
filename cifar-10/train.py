import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from tqdm import tqdm

from model import DecoderViT, EncoderViT

# 모델, optimizer, loss function 정의
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

PATCH_SIZE = 4
IN_CHANNELS = 3
EMBED_DIM = 512
NUM_HEADS = 6
NUM_LAYERS = 8
FFN_DIM = 512
DROPOUT = 0.1
NUM_CLASSES = 10

print(f"[Configurations]")
print(f"Using device: {device}")
print(f"Patch size: {PATCH_SIZE} x {PATCH_SIZE}")
print(f"Embedding dimension: {EMBED_DIM}")
print(f"Number of heads: {NUM_HEADS}")
print(f"Number of layers: {NUM_LAYERS}")
print(f"Feedforward dimension: {FFN_DIM}")
print(f"Dropout rate: {DROPOUT}")
print(f"Number of classes: {NUM_CLASSES}")
print()

# Decoder-based Vision Transformer 모델
model_decoder = DecoderViT(
    image_sizes=[32, 16, 8],
    patch_size=PATCH_SIZE,
    num_classes=NUM_CLASSES,
    dim=EMBED_DIM,
    depth=NUM_LAYERS,
    heads=NUM_HEADS,
    mlp_dim=FFN_DIM,
    channels=IN_CHANNELS,
    dropout=DROPOUT,
).to(device)

# Encoder-based Vision Transformer 모델
model_encoder = EncoderViT(
    image_size=32,
    patch_size=PATCH_SIZE,
    num_classes=NUM_CLASSES,
    dim=EMBED_DIM,
    depth=NUM_LAYERS,
    heads=NUM_HEADS,
    mlp_dim=FFN_DIM,
    channels=IN_CHANNELS,
    dropout=DROPOUT,
).to(device)

# 모델 파라미터 수 계산 및 출력
num_params_decoder = sum(p.numel() for p in model_decoder.parameters() if p.requires_grad)
num_params_encoder = sum(p.numel() for p in model_encoder.parameters() if p.requires_grad)
print(f"Number of trainable parameters (Decoder): {num_params_decoder:,d}")
print(f"Number of trainable parameters (Encoder): {num_params_encoder:,d}")

optimizer_decoder = optim.AdamW(model_decoder.parameters(), lr=1e-4, weight_decay=1e-5)
optimizer_encoder = optim.AdamW(model_encoder.parameters(), lr=1e-4, weight_decay=1e-5)

scheduler_decoder = optim.lr_scheduler.CosineAnnealingLR(optimizer_decoder, T_max=100)
scheduler_encoder = optim.lr_scheduler.CosineAnnealingLR(optimizer_encoder, T_max=100)

criterion = nn.CrossEntropyLoss()

# 데이터 로드 및 전처리
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_dataset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

def create_image_pyramid(image):
    """
    CIFAR-10 이미지(32x32x3)로부터 Image Pyramid를 생성합니다.
    Args:
        image: torch.Tensor, (C, H, W) 형태의 CIFAR-10 이미지
    Returns:
        list: Image Pyramid (3단계: 8x8x3, 16x16x3, 32x32x3)
    """
    pyramid = []
    for i in range(3):
        scale_factor = 2 ** i
        pooled_image = nn.functional.avg_pool2d(image, kernel_size=scale_factor, stride=scale_factor)
        pyramid.append(pooled_image)
    return pyramid

# 학습 루프
def train(model, device, train_loader, optimizer, criterion, is_decoder=True):
    model.train()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))

    num_data = 0
    sum_loss = 0
    correct = 0

    for batch_idx, (data, target) in pbar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        if is_decoder:
            # Image Pyramid 생성
            pyramid = create_image_pyramid(data)
            pyramid = [p.to(device) for p in pyramid]
            output = model(pyramid)

            # 모든 출력에 대한 loss 계산
            loss = 0
            for i in range(output.shape[1]):
                loss += criterion(output[:, i, :], target)
            loss /= output.shape[1]
        else:
            output = model(data)
            loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        # accuracy는 마지막 토큰의 accuracy만 계산
        sum_loss += loss.item() * data.size(0)
        if is_decoder:
            pred = output[:, -1, :].argmax(dim=1, keepdim=True)
        else:
            pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        num_data += data.size(0)

        avg_loss = sum_loss / num_data
        accuracy = 100. * correct / num_data

        pbar.set_description(f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f} %")



# 평가 루프 (Decoder 모델에 대해 토큰 위치별 평균 loss와 accuracy 계산 및 출력)
def test(model, device, test_loader, criterion, is_decoder=True):
    model.eval()
    test_loss = 0
    correct = 0

    if is_decoder:
        # Decoder 모델의 경우, 토큰 위치별 loss와 accuracy를 누적할 리스트
        token_losses = [0] * model.total_patches  # 모든 토큰 위치에 대한 loss를 저장할 리스트 초기화
        token_corrects = [0] * model.total_patches # 모든 토큰 위치에 대한 correct를 저장할 리스트 초기화
        token_counts = [0] * model.total_patches  # 모든 토큰 위치에 대한 data 개수를 저장할 리스트 초기화

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            if is_decoder:
                # Image Pyramid 생성
                pyramid = create_image_pyramid(data)
                pyramid = [p.to(device) for p in pyramid]
                output = model(pyramid)  # (batch_size, num_patches, num_classes)

                # 모든 토큰에 대한 loss와 accuracy 계산
                for i in range(output.shape[1]):
                    token_losses[i] += criterion(output[:, i, :], target).item() * data.size(0)
                    pred = output[:, i, :].argmax(dim=1, keepdim=True)
                    token_corrects[i] += pred.eq(target.view_as(pred)).sum().item()
                    token_counts[i] += data.size(0) # 해당 위치 토큰을 본 data 개수
            else:
                output = model(data)
                test_loss += criterion(output, target).item() * data.size(0)  # 배치 사이즈 곱하기
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

    if is_decoder:
        # Decoder 모델의 경우, 토큰 위치별 평균 loss와 accuracy 계산 및 출력
        token_avg_losses = [loss / count for loss, count in zip(token_losses, token_counts)]
        token_avg_accuracies = [100. * correct / count for correct, count in zip(token_corrects, token_counts)]

        print(f"Decoder Test Loss and Accuracy (per each token position):")
        for i in range(0, model.total_patches, 4):
            avg_loss = sum(token_avg_losses[i:i + 4]) / 4
            avg_accuracy = sum(token_avg_accuracies[i:i + 4]) / 4
            print(f"- Tokens {i + 1}-{i + 4}: Avg. Loss: {avg_loss:.4f}, Avg. Accuracy: {avg_accuracy:.2f}%")

        return token_avg_losses[-1], token_avg_accuracies[-1]  # 마지막 토큰의 loss와 accuracy 반환
    else:
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f"Encoder Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")

        return test_loss, accuracy

# 학습 및 평가
for epoch in range(1, 101):
    print(f"Epoch: {epoch}")
    print("Training Decoder...")
    train(model_decoder, device, train_loader, optimizer_decoder, criterion, is_decoder=True)
    print("Testing Decoder...")
    test_loss_decoder, accuracy_decoder = test(model_decoder, device, test_loader, criterion, is_decoder=True)
    scheduler_decoder.step()
    print(f"Decoder - Epoch: {epoch}, Test Loss: {test_loss_decoder:.4f}, Accuracy: {accuracy_decoder:.2f}%")

    print("Training Encoder...")
    train(model_encoder, device, train_loader, optimizer_encoder, criterion, is_decoder=False)
    print("Testing Encoder...")
    test_loss_encoder, accuracy_encoder = test(model_encoder, device, test_loader, criterion, is_decoder=False)
    scheduler_encoder.step()
    print(f"Encoder - Epoch: {epoch}, Test Loss: {test_loss_encoder:.4f}, Accuracy: {accuracy_encoder:.2f}%")