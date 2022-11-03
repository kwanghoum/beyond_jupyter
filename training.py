from models import model

def train_model(train_generator, valid_generator, epoch):
    model.fit(train_generator,
            validation_data=valid_generator,
            epochs=epoch)

def eval_model(test_generator):
    _, acc = model.evaluate(test_generator)

    return acc





        

        # # Graph
        # plt.figure(figsize=(8,5))
        # plt.plot(train_loss_list)
        # plt.plot(test_loss_list)
        # plt.legend(['Train Loss', 'Test Loss'])
        # plt.savefig('./results/' + self.model_name + '_graph.png')

        # self.model.load_state_dict(torch.load('./results/' + self.model_name + '_best.pth'))
        # train_acc = eval(self.trainloader)
        # test_acc = eval(self.testloader)
        # print(f'Epoch{best_epoch}: Train Accuracy: {train_acc}, Test Accuracy: {test_acc}')



    