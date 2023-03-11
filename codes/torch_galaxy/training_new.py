from http.client import IM_USED
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

###########################################################################
class CosineDecaySchedulerLR:

    def __init__(self, optimizer, base_lr, epochs, niter_epoch, initial_lr = 1e-10, final_lr = 1e-6, warmup_epochs = 0, constant_epochs = 0):
        # Store optimizer
        self.optimizer = optimizer
        # Save params
        self.params = {}
        self.params['base_lr']         = base_lr
        self.params['final_lr']        = final_lr
        self.params['initial_lr']      = initial_lr
        self.params['epochs']          = epochs
        self.params['niter_epoch']     = niter_epoch
        self.params['warmup_epochs']   = warmup_epochs
        self.params['constant_epochs'] = constant_epochs
        # Calcualte lr values
        self.lr_values = self.cosine_lr_values(**self.params)
        # Initialize iter counter
        self.it = -1

    def cosine_lr_values(self, base_lr, initial_lr, final_lr, epochs, niter_epoch, warmup_epochs, constant_epochs):
        # Get total amount of
        warmup_iters  = warmup_epochs * niter_epoch
        constant_iter = constant_epochs * niter_epoch
        decay_iter    = (epochs - (warmup_epochs + constant_epochs)) * niter_epoch
        # Check values
        assert epochs - (warmup_epochs + constant_epochs) >= 0, 'warmup_epochs + constant_epochs cannot be greater than epochs.'
        # Warmup lr values
        if warmup_iters > 0:
            warmup_lr = np.linspace(initial_lr, base_lr, warmup_iters)
        else:
            warmup_lr = np.array([])
        # Constant lr values
        if constant_iter > 0:
            constant_lr = np.array(constant_iter * [base_lr])
        else:
            constant_lr = np.array([])
        # Decay lr values
        if  decay_iter > 0:
            decay_lr = [final_lr + 0.5 * (base_lr - final_lr) * (1 + np.cos(np.pi * i / (decay_iter))) for i in range(1, decay_iter + 1)]
        else:
            decay_lr = np.array([])
        # Put together all values
        total_lr = np.concatenate((warmup_lr, constant_lr, decay_lr))

        return total_lr

    def step(self):
        # Update iteration
        self.it = self.it + 1
        # Update lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr_values[self.it]
            #print(self.lr_values[self.it], self.it)         

###########################################################################

def print_performance(epoch, loss_train, acc_train, loss_valid, acc_valid, round_to = 3):
    print('Epoch: ' + str(epoch + 1) + ' || Train loss = ' + str(round(loss_train, round_to)) + ', Train Acc = ' + str(round(acc_train, round_to)) + ', Valid Loss = ' + str(round(loss_valid, round_to)) + ', Valid Acc = ' + str(round(acc_valid, round_to)))
    return None

def get_labels(network_output):
    _, pred_labels = torch.max(network_output, dim = 1)
    return pred_labels

def get_correct_predictions(pred_labels, labels):
    correct_preds = torch.sum(pred_labels == labels).item()
    return correct_preds

###########################################################################
# VALID

def merge_eval_results(all_names, all_labels, all_outputs, soft_max = torch.nn.Softmax(dim = 1)):

    # Put together names and real labels
    names_labels = {'img': all_names, 
                    'true_label': all_labels.to('cpu').numpy()}

    names_labels = pd.DataFrame(names_labels)

    # Apply softmax to net outputs
    proba = soft_max(all_outputs).to('cpu').numpy()
    col_names = ['p_' + str(i) for i in range(proba.shape[1])]
    proba = pd.DataFrame(proba, columns = col_names)

    # Add probabilities to each point
    eval_results = pd.concat([names_labels, proba], axis = 1)
    # Get mean probability por each class
    eval_results = eval_results.groupby(['img', 'true_label'], as_index = False, sort = False).agg('mean')
    # Get predicted class
    eval_results['pred_label'] = eval_results[col_names].apply(lambda row: row.argmax(), axis = 1)

    return eval_results

def evaluate_model(network, loss_fn, data_loader, device, return_pred = False):

    # Turn on evlautaion mode 
    network.eval()
    
    with torch.no_grad():
        # Get total amount of batches
        total_batches = len(data_loader)
        # Initialize total loss and total acc
        total_loss = 0
        # Create a empty tensor to store the network's outputs
        all_outputs = torch.tensor([], device = device)
        # Create empty lists to store names and labels
        all_names = []
        all_labels = torch.tensor([], device = device)

        # Iterate over data
        for batch in data_loader:
            # Get data
            names, imgs, labels = batch
            # Send data to device
            imgs, labels = imgs.to(device), labels.to(device)
            # Get the netwrok's outputs and evaluate
            # the batch loss
            outputs = network(imgs)
            loss = loss_fn(outputs, labels)
            # Update loss
            total_loss = total_loss + loss.item()
            # Store batch info
            all_outputs = torch.cat((all_outputs, outputs))
            all_labels = torch.cat((all_labels, labels))
            all_names = all_names + list(names)
            
        # Get average loss across batches
        total_loss = total_loss / total_batches
        # Calculate most voted class for each image
        eval_results = merge_eval_results(all_names, all_labels, all_outputs)
        # Calculate accuracy
        total_acc = accuracy_score(eval_results.true_label, eval_results.pred_label)

        # Return all predictions if asked
        if return_pred:
            return total_loss, total_acc, eval_results
        else:
            return total_loss, total_acc

###########################################################################

def train_one_epoch(network, loss_fn, optimizer, data_loader, device, lr_scheduler = None):

    # Turn on train mode
    network.train()
    # Get total amount of batches and
    # total amount of images
    total_batches = len(data_loader)
    total_images  = len(data_loader.dataset)
    # Initialize total loss and total acc
    total_loss = 0
    total_acc  = 0

    # Iterate over data
    for batch in data_loader:
        # Get data    
        imgs, labels = batch
        # Send data to device
        imgs, labels = imgs.to(device), labels.to(device)
        # Update lr
        if lr_scheduler:
            lr_scheduler.step()
        # Train the network
        optimizer.zero_grad()
        outputs = network(imgs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        # UPDATE STATISTICS
        # Loss
        total_loss = total_loss + loss.item()
        # Accuracy
        pred_labels = get_labels(outputs)
        total_acc = total_acc + get_correct_predictions(pred_labels, labels)
        
    # Get average loss across batches
    total_loss = total_loss / total_batches
    # Calcualte overall accuracy
    total_acc = total_acc / total_images

    return total_loss, total_acc

###########################################################################

def train_network(epochs, network, optimizer, loss_fn, train_dataloader, valid_dataloader, device, early_stopping = np.Inf, lr_scheduler = None):
    
    # Keeps track of the best model's performance
    best_accuracy = - np.inf
    best_loss     = None
    # Start early stopping counter
    es_count = 0
    # Check scheduler and trainer epochs
    if lr_scheduler:
        assert lr_scheduler.params['epochs'] == epochs, 'lr_scheduler epochs and train_network epochs must be equal.'
    # Send network to device
    network = network.to(device)

    # Train the number of epochs
    for epoch in range(epochs):
        # Train the model
        loss_train, acc_train = train_one_epoch(network      = network, 
                                                loss_fn      = loss_fn, 
                                                optimizer    = optimizer, 
                                                data_loader  = train_dataloader,
                                                device       = device,
                                                lr_scheduler = lr_scheduler)
        # Evaluate model
        loss_valid, acc_valid = evaluate_model(network     = network, 
                                               loss_fn     = loss_fn, 
                                               data_loader = valid_dataloader,
                                               device      = device)
        # Saves the best model
        if best_accuracy < acc_valid:
            #torch.save(network.state_dict(), 'best_weights.pt')
            # Update best performance
            best_accuracy = acc_valid 
            best_loss     = loss_valid
            # Reset es_count
            es_count = 0   
        else:
            # Updates early stopping counter
            es_count = es_count + 1

        # Shows progress
        print_performance(epoch, loss_train, acc_train, loss_valid, acc_valid)

        # EARLY STOPPING
        if es_count >= early_stopping:
            print('EARLY STOPPING!!!!')
            print('BEST PERFORMANCE \nVALID ACC = '  + str(round(best_accuracy, 4)) + '\nVALID LOSS ' + str(round(best_loss, 4)))
            return best_loss, best_accuracy

    print('FINISHED TRAINIG, HURRAY! :D')
    print('BEST PERFORMANCE \nVALID ACC = '  + str(round(best_accuracy, 4)) + '\nVALID LOSS ' + str(round(best_loss, 4)))

    return best_loss, best_accuracy