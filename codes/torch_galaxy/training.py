from jsonschema import RefResolutionError
import torch
import pandas as pd
import numpy as np

def print_performance(epoch, loss_train, acc_train, loss_valid, acc_valid, round_to = 3):
    print('Epoch: ' + str(epoch + 1) + ' || Train loss = ' + str(round(loss_train, round_to)) + ', Train Acc = ' + str(round(acc_train, round_to)) + ', Valid Loss = ' + str(round(loss_valid, round_to)) + ', Valid Acc = ' + str(round(acc_valid, round_to)))
    return None

def get_labels(network_output):
    _, pred_labels = torch.max(network_output, dim = 1)
    return pred_labels

def get_correct_predictions(pred_labels, labels):
    correct_preds = torch.sum(pred_labels == labels).item()
    return correct_preds

def one_epoch(network, loss_fn, data_loader, optimizer = None, train = False, return_pred = False):

    # Get total amount of batches and
    # total amount of images
    total_batches = len(data_loader)
    total_images  = len(data_loader.dataset)
    # Initialize total loss and total acc
    total_loss = 0
    total_acc  = 0
    # Create a empty tensor to store the predicted labels
    if return_pred:
        all_preds = torch.tensor([], dtype = torch.int)

    # Iterate over data
    for batch in data_loader:

        # Get data    
        imgs, labels = batch

        if train:
            # Train the network
            optimizer.zero_grad()
            outputs = network(imgs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
        else:
            # Just get the outputs for validation
            outputs = network(imgs)
            loss = loss_fn(outputs, labels)

        # UPDATE STATISTICS
        # Loss
        total_loss = total_loss + loss.item()
        # Accuracy
        pred_labels = get_labels(outputs)
        total_acc = total_acc + get_correct_predictions(pred_labels, labels)
        
        # Stores the predicted lables
        if return_pred:
            all_preds = torch.cat((all_preds, pred_labels))

    # Get average loss across batches
    total_loss = total_loss / total_batches
    # Calcualte overall accuracy
    total_acc = total_acc / total_images

    # Return predicted labels or just loss and acc
    if return_pred:
        return total_loss, total_acc, all_preds
    else:
        return total_loss, total_acc

def train_one_epoch(network, loss_fn, optimizer, data_loader):
    network.train()
    train_loss, train_acc = one_epoch(network     = network, 
                                      loss_fn     = loss_fn, 
                                      optimizer   = optimizer,
                                      data_loader = data_loader, 
                                      train       = True)
    return train_loss, train_acc

def evaluate_model(network, loss_fn, data_loader, return_pred = False):
    
    network.eval()
    with torch.no_grad():    
        if return_pred:
            eval_loss, eval_acc, pred_labels = one_epoch(network     = network, 
                                                         loss_fn     = loss_fn, 
                                                         data_loader = data_loader, 
                                                         train       = False,
                                                         return_pred = True) 
            return eval_loss, eval_acc, pred_labels
        else:
            eval_loss, eval_acc = one_epoch(network     = network, 
                                            loss_fn     = loss_fn, 
                                            data_loader = data_loader, 
                                            train       = False)
            return eval_loss, eval_acc

def train_network(epochs, network, optimizer, loss_fn, train_dataloader, valid_dataloader):

    # Keeps track of the best model's performance
    best_accuracy = - np.inf
    best_loss     = None

    for epoch in range(epochs):

        loss_train, acc_train = train_one_epoch(network     = network, 
                                                optimizer   = optimizer, 
                                                loss_fn     = loss_fn, 
                                                data_loader = train_dataloader)

        loss_valid, acc_valid = evaluate_model(network     = network, 
                                            loss_fn     = loss_fn, 
                                            data_loader = valid_dataloader)

        # Saves the best model
        if best_accuracy < acc_valid:
            torch.save(network.state_dict(), 'best_weights.pt')
            best_accuracy = acc_valid 
            best_loss     = loss_valid   
        
        print_performance(epoch, loss_train, acc_train, loss_valid, acc_valid)
    
    print('FINISHED TRAINIG, HURRAY! :D')
    print('BEST PERFORMANCE \nVALID ACC = '  + str(round(best_accuracy, 3)) + '\nVALID LOSS ' + str(round(best_loss, 3)))

    return best_accuracy, best_accuracy