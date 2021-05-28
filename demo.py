import os
from ocr import ocr
import time
import shutil
import numpy as np
from PIL import Image
from glob import glob

def single_pic_proc(image_file):
    image = np.array(Image.open(image_file).convert('RGB'))
    result, image_framed = ocr(image)
    return result,image_framed


if __name__ == '__main__':
    image_files = glob('./test_images/*.*')
    result_dir = './test_result'
    for image_file in sorted(image_files):
        t = time.time()
        result, image_framed = single_pic_proc(image_file)
        output_file = os.path.join(result_dir, image_file.split('/')[-1])
        txt_file = os.path.join(result_dir, image_file.split('/')[-1].split('.')[0]+'.txt')
        
        print(txt_file)
        txt_f = open(txt_file, 'w')
        Image.fromarray(image_framed).save(output_file)
        print("Mission complete, it took {:.3f}s".format(time.time() - t))
        print("\nRecognition Result:\n")
        for key in result:
            print(result[key][1])
            txt_f.write(result[key][1]+'\n')
        txt_f.close()

def train(train_loader, model, criteon, optimizer, epoch):
    train_loss = 0
    train_acc = 0
    num_correct= 0
    for step, (x,y) in enumerate(train_loader):

        # x: [b, 3, 224, 224], y: [b]
        x, y = x.to(device), y.to(device)

        model.train()
        logits = model(x)
        loss = criteon(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += float(loss.item())
        train_losses.append(train_loss)
        pred = logits.argmax(dim=1)
        num_correct += torch.eq(pred, y).sum().float().item()
    logger.info("Train Epoch: {}\t Loss: {:.6f}\t Acc: {:.6f}".format(epoch,train_loss/len(train_loader),num_correct/len(train_loader.dataset)))
    return num_correct/len(train_loader.dataset), train_loss/len(train_loader)