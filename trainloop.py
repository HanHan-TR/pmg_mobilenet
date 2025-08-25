import torch
from utils import jigsaw_generator, cosine_anneal_schedule


def train_loop(fold_k,
               epochs_fold,
               nb_epoch,
               train_loader,
               val_loader,
               model,
               optimizer,
               lr,
               loss_fn,
               batch_size=24,
               device='cuda'):
    for iter in range(epochs_fold):
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_loss3 = 0
        train_loss4 = 0
        correct = 0
        total = 0
        idx = 0

        epoch = fold_k * epochs_fold + iter
        for nlr in range(len(optimizer.param_groups)):
            optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, nb_epoch, lr[nlr])

        # ------------------------------------- one dataset loop ----------------------------------------
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            idx = batch_idx

            if inputs.shape[0] < batch_size:
                continue

            inputs, targets = inputs.to(device), targets.to(device)

            # update learning rate

            # Step 1
            optimizer.zero_grad()
            inputs1 = jigsaw_generator(inputs, 8)  # 图像尺寸缩小8倍，图像分割为8*8个块，块越多，尺寸下采样的程度越小
            output_1, _, _, _ = model(inputs1)
            loss1 = loss_fn(output_1, targets) * 1
            loss1.backward()
            optimizer.step()

            # Step 2
            optimizer.zero_grad()
            inputs2 = jigsaw_generator(inputs, 4)
            _, output_2, _, _ = model(inputs2)
            loss2 = loss_fn(output_2, targets) * 1
            loss2.backward()
            optimizer.step()

            # Step 3
            optimizer.zero_grad()
            inputs3 = jigsaw_generator(inputs, 2)
            _, _, output_3, _ = model(inputs3)
            loss3 = loss_fn(output_3, targets) * 1
            loss3.backward()
            optimizer.step()

            # Step 4
            optimizer.zero_grad()
            _, _, _, output_concat = model(inputs)
            concat_loss = loss_fn(output_concat, targets) * 2
            concat_loss.backward()
            optimizer.step()

            #  training log
            _, predicted = torch.max(output_concat.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            train_loss += (loss1.item() + loss2.item() + loss3.item() + concat_loss.item())
            train_loss1 += loss1.item()
            train_loss2 += loss2.item()
            train_loss3 += loss3.item()
            train_loss4 += concat_loss.item()

            if batch_idx % 10 == 0:
                print(
                    'Step: %d | Loss1: %.3f | Loss2: %.5f | Loss3: %.5f | Loss_concat: %.5f | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                        batch_idx, train_loss1 / (batch_idx + 1), train_loss2 / (batch_idx + 1),
                        train_loss3 / (batch_idx + 1), train_loss4 / (batch_idx + 1), train_loss / (batch_idx + 1),
                        100. * float(correct) / total, correct, total))

        # --------------------------- end of one dataset loop -------------------------------------------

        train_acc = 100. * float(correct) / total
        train_loss = train_loss / (idx + 1)
        with open(str(exp_dir) + '/results_train.txt', 'a') as file:
            file.write(
                'Iteration %d | train_acc = %.5f | train_loss = %.5f | Loss1: %.3f | Loss2: %.5f | Loss3: %.5f | Loss_concat: %.5f |\n' % (
                    epoch, train_acc, train_loss, train_loss1 / (idx + 1), train_loss2 / (idx + 1), train_loss3 / (idx + 1),
                    train_loss4 / (idx + 1)))

        val_acc, val_acc_com, val_loss = test(net, CELoss, batch_size, val_root)
        if val_acc_com > max_val_acc:
            max_val_acc = val_acc_com
            # model_name="/model_epoch{}.pt".format(epoch)
            state = {'epoch': epoch,
                     'model': net.state_dict(),
                     'accuracy': val_acc_com,
                     }
            model_name = "/best.pth"
            torch.save(state, str(exp_dir) + model_name)
            with open(str(exp_dir) + '/results_test.txt', 'a') as file:
                file.write('Iteration  %d, test_acc = %.5f, test_acc_combined = %.5f, test_loss = %.6f\n' % (epoch, val_acc, val_acc_com, val_loss))
