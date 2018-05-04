#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 15:44:57 2018

@author: xavipor

based on  Ryan Harris videos (https://youtu.be/Ku7D-F6xOUM?list=PLRyu4ecIE9tibdzuhJr94uQeKnOFkkbq6)

It is think to work as W*x (colum)
because of this from the first layer to the first neuro to the second layer we have:
    
    W11 W21 W31 W41 * colum(x1 x2 x3 1)
    
And also note that if we go from a layer with m nodes to n. W will be 

n * m+1 because of the bias. 

****************************************
work without batches, just with one column of data (one example) per time, can be done
but we need to loop over all the examples. 

Instead of that we can think about doing all in just one pass if we do the following. We will need to build
a matrix of data Data=(oneColumnExample|secondColumnExample.....|lastColumnExample) 
and we can just multiply as before W*Data = (W*oneColumnExample|....|W*lastColumnExample)


"""



import numpy as np

class BackPropagationNetwork:
    
    layerCount=0
    shape=None
    weights=[]
    
    #Methods
    
    def __init__(self,layerSize):
        
        
        
        #Layer info
        self.layerCount=len(layerSize)-1  #If you pass (2,2,1) only need 2 layers of weights 
        self.shape=layerSize
        
        #input/Output data from last run
        self._layerInput=[]
        self._layerOutput=[]
        self._previousWeightDelta=[]
        
        #Create weight arrays. Be aware that the weights will have as many rows
        #as the "next layer" and columns as the "current layer +1 bias"
        
        for (l1,l2) in zip (layerSize[:-1],layerSize[1:]):
            self.weights.append(np.random.normal(scale=0.1,size=(l2,l1+1))) #plus 1 because of the Bias
            self._previousWeightDelta.append(np.zeros((l2,l1+1)))
        
    #Run method
    def Run(self,input):
        """Run the network given input data, in rows. It will ve transpose in here
        to work with data in columns"""
        
        #each row is an example with d column features
        lnCases=input.shape[0]
        
        #Clear the previous intermediate lists
        self._layerInput=[]
        self._layerOutput=[]
        
        
        #Run
        
        for index in range(self.layerCount):
            if index==0:          

                #Here first row of weights are the wights from the first layer to the first neuron of the second layer
                #They are going to be multiplied for the first example (colum of np.vstack(...))
                layerInput=self.weights[0].dot(np.vstack([input.T,np.ones([1,lnCases])])) #vstack just to add the biases
            else:

                layerInput=self.weights[index].dot(np.vstack([self._layerOutput[-1],np.ones([1,lnCases])]))
            #Each column is an example of each example that pass trough that layer. It is saved in an array of arrays.
            #Each row is the value of the hidden layers for each example. 
            self._layerInput.append(layerInput)
            #The same for output, it is just after squish the input through the function.
            self._layerOutput.append(self.sgm(layerInput))
            
        return self._layerOutput[-1].T #Just to return a row per example and a column per neuron (in this case just one neuron at the output
        
    def TrainEpoch(self,input,target,trainingRate=0.2,momentum=0.5):
        """Training the network for one epoch"""
        delta = []#Each column an example
        lnCases= input.shape[0]
        
        #First of all run the network
        
        self.Run(input)
        
        #Calculate the deltas. Starting in the last output layer. 
        
        for index in reversed(range(self.layerCount)):
            if index == self.layerCount - 1:
                #Compare to the target values
                output_delta= self._layerOutput[index] -target.T #get the errors (1x4). Each column an example
                error= np.sum(output_delta**2)
                #Note that this is element wise not dotProduct
                delta.append(output_delta*self.sgm(self._layerInput[index],True)) #Derivative upstream * local derivative.
            else:
                #Compare to the following layerś delta.
                #We need to traspose the weights because now we need to multiply all the weigths from the layer L+1 (we are going backwards)
                #that happens to point to one neuron from the layer L. Something completely different that what we did
                #in the forward pass where we multiply all the weights pointing from the layer L to one neuron of layer L+1
                delta_pullback = self.weights[index+1].T.dot(delta[-1])#classic delta*weights (to have the typical summatory, all the deltas that affec)
                #Again the upstream Gradient * local gradient.
                #Element wise multiplication not dot product
                #not using the last row (biases)
                delta.append(delta_pullback[:-1,:] * self.sgm(self._layerInput[index],True)) #The minus -1 it is because of the BIAS and we do not need the BIAS for backprop
        
        #Compute weight deltas (Activation * deltas)
        for index in range(self.layerCount):
            delta_index= self.layerCount -1 - index 
            
            if index==0:
                layerOutput=np.vstack([input.T,np.ones([1,lnCases])])
            else:
                layerOutput=np.vstack([self._layerOutput[index-1],np.ones([1,self._layerOutput[index-1].shape[1]])])
            #Collapsing the matrix of Deltaweights to update the weights. In each layer of the 3D tensor we have one example. We need to collapse the depth (all the examples)
            #Just converting the rows in colums and depth in examples. This way, the deltas are columns and the layeroutputs are rows           
            #(refer to the video     https://youtu.be/nOCaFkh4NSs?list=PLRyu4ecIE9ti5wsokn1j_ZJU7a7N5hREf    )
            
            #layerOutput[None,:,:].transpose(2,0,1)
            #With layerOutput[None,:,:] we get a cube, where for each layer of de depth we have a 2D array where
            #in each colum we have an example.
            #After .transpose(2,1,0)
            #We end up with a matrix where for each layer we have an example in a row.           
            
            
            #delta[delta_index][None,:,:].transpose(2,1,0)
            #With delta[delta_index][None,:,:] we get a cube, where for each layer of de depth we have a 2D array where
            #in each column we have an example.
            #After .transpose(2,1,0)
            #We end up with a matrix where for each layer we have an example in a cloumn. 
            
            #Each example lives in a layer and then we collapse in depth to sum.
            
            curWeightDelta= np.sum(layerOutput[None,:,:].transpose(2,0,1)*delta[delta_index][None,:,:].transpose(2,1,0),axis=0)
            weightDelta=trainingRate * curWeightDelta + momentum *self._previousWeightDelta[index]
            
            self.weights[index]-=  weightDelta
            
            self._previousWeightDelta[index]=weightDelta
        
        return error
            
            
             
              
        
            
    #Transfer functionsself._layerOutput
    def sgm(self,x,Derivative =False):
        if not Derivative:
            return 1/(1+np.exp(-x))
        else:
            out=self.sgm(x)
            return out*(1-out)
            
            
            
if __name__=="__main__":
    bpn =BackPropagationNetwork((2,2,1))
    lvInput=np.array([[0,0],[1,1],[0,1],[1,0]])
    lvTarget=np.array([[0.05],[0.05],[0.95],[0.95]])
    lnMax=100000000
    lnErr=1e-6
    
    for i in range (lnMax+1):
        err=bpn.TrainEpoch(lvInput,lvTarget,momentum=0.7)
        if i % 2500 ==0:
            print("Iteration {0} \t Error: {1:0.6f}".format(i,err))
            
        if err <= lnErr :
            print("Minimun error reached at iteration {0}".format(i))
            break
    lvOutput=bpn.Run(lvInput)
    print("Input: {0} \n nOuutput: {1}".format(lvInput,lvOutput))
    
    
        
    