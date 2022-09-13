import numpy
import scipy.io
import math
import geneNewData
import statistics

def main():
    myID='8816'
    geneNewData.geneData(myID)
    Numpyfile0 = scipy.io.loadmat('digit0_stu_train'+myID+'.mat')
    Numpyfile1 = scipy.io.loadmat('digit1_stu_train'+myID+'.mat')
    Numpyfile2 = scipy.io.loadmat('digit0_testset'+'.mat')
    Numpyfile3 = scipy.io.loadmat('digit1_testset'+'.mat')
  
    train0 = Numpyfile0.get('target_img')
    train1 = Numpyfile1.get('target_img')
    test0 = Numpyfile2.get('target_img')
    test1 = Numpyfile3.get('target_img')
    print([len(train0),len(train1),len(test0),len(test1)])
    print('Your trainset and testset are generated successfully!')
    
    # calculate mean and standard deviation of each picture in train dataset
    train0_avg = numpy.empty([5000,1])
    train0_std = numpy.empty([5000,1])
    train1_avg = numpy.empty([5000,1])
    train1_std = numpy.empty([5000,1])
    for i in range(5000):
        train0_avg[i]= numpy.average(train0[i])
        train0_std[i]= numpy.std(train0[i])
        #print('Average of train0: ',i, train0_avg[i])
        #print('Std dev of train0: ',i, train0_std[i])
        train1_avg[i]= numpy.average(train1[i])
        train1_std[i]= numpy.std(train1[i])
    parr1 = numpy.mean(train0_avg)
    print("\nMean-train0_avg: ", parr1)
    parr2 = numpy.var(train0_avg)
    print("\nvariance-train0_avg: ", parr2)
    parr3 = numpy.mean(train0_std)
    print("\nMean-train0_std: ", parr3)
    parr4 = numpy.var(train0_std)
    print("\nvariance-train0_std: ", parr4)
    parr5 = numpy.mean(train1_avg)
    print("\nMean-train1_avg: ", parr5)
    parr6 = numpy.var(train1_avg)
    print("\nvariance-train1_avg: ", parr6)
    parr7 = numpy.mean(train1_std)
    print("\nMean-train1_std: ", parr7)
    parr8 = numpy.var(train1_std)
    print("\nvariance-train1_std: ", parr8)
    
     # calculate mean and standard deviation of each picture in test dataset
    test0_avg = numpy.empty([980,1])
    test0_std = numpy.empty([980,1])
    for i in range(980):
        test0_avg[i]= numpy.average(test0[i])
        test0_std[i]= numpy.std(test0[i])
        #print('Average of test0: ',i, test0_avg[i])
        #print('Std dev of test0: ',i, test0_std[i])
    test1_avg = numpy.empty([1135,1])
    test1_std = numpy.empty([1135,1])
    for i in range(1135):
        test1_avg[i]= numpy.average(test1[i])
        test1_std[i]= numpy.std(test1[i])
        #print('Average of test1: ',i, test1_avg[i])
        #print('Std dev of test1: ',i, test1_std[i])
     
    # NB Classifier
    def NB_probability(xi, mean, variance):
        std = math.sqrt(variance)
        expo = math.exp(-((xi-mean)**2 / (2 * std**2 )))
        return (1 / (math.sqrt(2 * math.pi) * std)) * expo
    
    #print(NB_probability(1.0, 1.0, 1.0))
    def decide0(test0avg_nparray,test0std_nparray ,parameter1, parameter2, parameter3,parameter4,parameter5, parameter6, parameter7,parameter8):
        test0_result= numpy.empty([980,1])
        test0_prob = numpy.empty([980,1])
        test1_prob = numpy.empty([980,1])
        for i in range(0,len(test0avg_nparray)):
            prob0 = 0.5*NB_probability(test0avg_nparray[i],parameter1, parameter2)*NB_probability(test0std_nparray[i],parameter3, parameter4)
            test0_prob[i] = prob0
            #print(i) 
            #print(test0_prob[i])
        for i in range(0,len(test0avg_nparray)):
            prob1 = 0.5*NB_probability(test0avg_nparray[i],parameter5, parameter6)*NB_probability(test0std_nparray[i],parameter7, parameter8)
            test1_prob[i] = prob1 
        for i in range(0,len(test0_prob)):
            if test0_prob[i]>test1_prob[i]:
                test0_result[i] = 0
            else:
                test0_result[i] =1
        #print(test0_result)
        accuracy = (test0_result == 0).sum() / 980
        print('accuracy of 0:', accuracy)
        
    def decide1(test1avg_nparray,test1std_nparray ,parameter1, parameter2, parameter3,parameter4,parameter5, parameter6, parameter7,parameter8):
        test1_result= numpy.empty([1135,1])
        test0_prob = numpy.empty([1135,1])
        test1_prob = numpy.empty([1135,1])
        for i in range(0,len(test1avg_nparray)):
            prob0 = 0.5*NB_probability(test1avg_nparray[i],parameter1, parameter2)*NB_probability(test1std_nparray[i],parameter3, parameter4)
            test0_prob[i] = prob0
            #print(i) 
            #print(test1_prob[i])
        for i in range(0,len(test1avg_nparray)):
            prob1 = 0.5*NB_probability(test1avg_nparray[i],parameter5, parameter6)*NB_probability(test1std_nparray[i],parameter7, parameter8)
            test1_prob[i] = prob1 
        for i in range(0,len(test0_prob)):
            if test1_prob[i]>test0_prob[i]:
                test1_result[i] = 1
            else:
                test1_result[i] =0
            #print(test1_result[i])
        accuracy = (test1_result == 1).sum() / 1135
        print('accuracy of 1:', accuracy)    
    
    decide0(test0_avg,test0_std,parr1,parr2,parr3,parr4,parr5,parr6,parr7,parr8)
    decide1(test1_avg,test1_std,parr1,parr2,parr3,parr4,parr5,parr6,parr7,parr8)
    
    
    pass


if __name__ == '__main__':
    main()