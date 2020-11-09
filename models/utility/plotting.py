import matplotlib.pyplot as plt

#def plot_confusion_matrix(cfm, percent=False):
    #plt.gca().matshow(cfm, cmap=plt.cm.Blues)
    
    #def number_to_str(num, percent):
        #if percent:
            #return "{:.3f}".format(num*100) + "%"
        #else:
            #return num
    
    #for i in range(cfm.shape[1]):
        #for j in range(cfm.shape[0]):
            #plt.gca().text(i,j,number_to_str(cfm[j,i], percent))
            
    #plt.xlabel("predicted values")
    #plt.ylabel("true values")
    
    #plt.show()
    
def plot_history(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
