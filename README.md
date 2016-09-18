AutoEncoder
===========

##Overview

This is a simple AutoEncoder program without using "Deep Learning FrameWork". 
(But use only sklearn to get MNIST DATA.)

I've been studying an autoencoder program in python since July 2016. 
I wanted to create a program that does not use a deep learning framework. 
Because using the framework, we will not know the internal operation.
I wanted to do a variety of experiments in this program.

This program was created to cut and paste from many tutorials and samples.
Only report part was made by myself (and it's so complicated).

## Description

  1. This program not use "Deep Learning FrameWork".
  1. This program operates comfortable alone CPU (not require GPU).
    (This program will give priority to speed, it has been at the expense of accuracy.)
  1. The experimental results are reported in the simple format with matplotlib.
  1. We can improve this program easily because it's a very simple.
  1. Example program encode/decode MNIST data, but core autoencoder class can use any purpose.

## Requirement
  * This program work at python 2 and 3.
  * I checked on Python 2.1.7 and 3.5.2.

  libraly is following
 * numpy 1.11.1
 * scipy 0.18.0
 * sklearn 0.17.1
 * PIL 2.0 / Pillow 3.3.1
 * matplotlib 1.5.3
 * pandas 0.18.1
 * (and jupyter)


## Usage
    python  Experiment.py

  or run on Jupyter or Python IDE.
  This program has not main() because I often execute this program on Jupyter Notebook.
  Jupyter Notebook is very convenient to experiment or have a try.

  Please run Experiment.py program, and it's prepared MNIST DATA and learn them and 
  save the results data, save the report image and show report of results on your display.

## Parameters and Report
### 0.Report Image

  ![AutoEncoderReport](https://github.com/np2lkoo/AutoEncoder/blob/master/ReportSample.JPG "Report")

### 1. Experiment condition and Information

| Index           | Function           | Comments                               |
|-----------------|:-------------------|:---------------------------------------|
| Node Count      |Node(Nuron)Count    |param:par_hidden:about from 10 to 1000                   |
| Activation Func |Activate function   |Sigmoid , ReLU and so on (can set in AutoEncoder.py) |
| Batch Size      |Learning size at the same time|param:par_batch_size:it must be Multiple of 10         |
| Noise Ratio     | -                  |param:par_noise:about from 0 to 0.9, recommend 0.3 to 0.4 |
| Epoch Limit     |Specify Epoch up to |param:par_epoch:about from 2\*\*1 to 2\*\*15                |
| DropOut         |perform the drop-out  Ratio|param:par_dropout:about from 0 to 0.9                    |
| Train Shuffle   |Shuffle order when learn|param:par_train_shuffle:not implimented yet                       |
| W untied        |Weight untied or not|param:par_untied:True:untied, False:tied                 |
| Alpha Ratio     |Leaning Ratio to  Update Weight | Fit Value to each Activation Function.(This value set by Activation Func in this program.)|
| Beta Ratio      |sparse regularize Ratio|not implimented yet.                      |
| Normalization    |-                   |not implimented yet. |
| Whitening       |Auto or ZCA or PCA  Whitening |not implimented yet. |
| Train SubScale  |thinning of learning |about from 6000 to 1. Larger as giving priority to speed and losing the accuracy. 1000 Epoch and 1/1000 subscale are almost Real 1 Epoch. Recommend from 100 to 1000|
| W Transport     |Report the way to W initialize|Use Learned Weight for initialized or not |
| Optimizer       |-                   |not implimented yet. SGD(stochastic gradient descent) only now         |
| Researcher      |-                   |Set your name                           |
| DateTime        |Experiment datetime |Experiment start datetime               |
| Framework       |use framework Name or not|:None only now                     |
| GPU/GPU         |-                   |put your favorite string (not auto-check)|
| Note            |-                   |func:.set_note(Note):Comment to Experiment(can set after the experiment)|

###2. Learning Term index and Range of W Weigth

| Index           | Function          | Comments                               |
|-----------------|:------------------|:---------------------------------------|
| Period          |MilseStone of Experiment|Period is each 2\*\*n on Epoch. About "n" is from 4 to 14. An intermediate situation saves and report on each Period. Epoch=2\*\*(Period -1)|
| (epoch)         |report Epoch count |Learn thinning by Train SubScale parameter. Then "1 epoch" does not mean to learn all of the data.|
| Wmax            |max range of Hidden Layer W |maximum value of the hidden layer W to be displayed on the right-hand side of this index|
|mean             |mean of Hidden Layer W |It should almost from -1.0 to 1.0, but there is also a case that will rapidly shift from zero when there is an error in the algorithm.|
|Wmin             |min of Hidden Layer W |maximum value of the hidden layer W to be displayed on the right-hand side of this index|

###3. W Weight Range（Display Range Graph）
 
| Index           | Function          | Comments                               |
|-----------------|:------------------|:---------------------------------------|
|W Range          |Graph of W Range   |to display max, mean, min by Error-Bar graph. Since the range of the vertical scale  is common, you can check changing (growing weight) of learning|
 
###4. W Weight (Display Gray Image)

| Index           | Function          | Comments                               |
|-----------------|:------------------|:---------------------------------------|
|W (Hidden Layer) |Gray scale display of the weights the hidden layer W |You can view images by using the Matplotlib of gray () function. W layer aria's "[number]" is a node number specified report function. You can set like 'w_select = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]'|

###5. Bias b (Display Range Graph and Value)

| Index           | Function          | Comments                               |
|-----------------|:------------------|:---------------------------------------|
| b range         |Graph of b Range   |to display bias b Error-Bar graph. Only selected 10 b values plot of left-hand side W belogs to. You can see all b values plot on No.12 Area Graph.|
| bmax            |max value of b in all Nodes|-|
| mean            |mean of b in all Nodes |-|
| bmin            |min value of b in all Nodes|-|

###6. processing time, Weight gain, and ealuation of decode image 

| Index           | Function          | Comments                               |
|-----------------|:------------------|:---------------------------------------|
| Time(sec)       |Execute time in Period.|for example Value of Period11 is Execute time from Epoch513 to Epoch1024|
| Wgain           |W entropy gain|Sum of Entropy gain of W Weight(from init befor to learned after)|
| Cost            |Error between Original data and Decode data |I call it "cost" that the diffence of Original image and Decode image. Because I learned "Neural Networks and Deep Learning" online book written by Michael Nielsen. He calls it the "Cost". |
| Img-diff        |the difference between Original and Decode|The value is square root of sum of each dot difference(that's the 256 gray scale)|
| dx-ent.         |Entropy diff of after decode from before encode|There is no probably strict. may be|
 
###7. x_hat(decode results) (Display Gray Image)

| Index           | Function          | Comments                               |
|-----------------|:------------------|:---------------------------------------|
| x_hat of Sample MNIST DATA & calibration|Display gray image|calculated When report x_hat image using learned W Weight|
 
###8. Decode display for zero and mean

| Index           | Function          | Comments                               |
|-----------------|:------------------|:---------------------------------------|
| Zero            | Flat Zero image   |encode/decode of zero array             |
| Flat Mean       | Flat Mean image   |encode/decode of mean array             |

###9. Display if MNIST original Image

| Index           | Function          | Comments                               |
|-----------------|:------------------|:---------------------------------------|
| X a sample MNIST data |Original image |Base data of encode                     |

###10. Display if MNIST original Image

| Index           | Function          | Comments                               |
|-----------------|:------------------|:---------------------------------------|
| Y training data |traning image      |same as X usually.But it can be different from X when W is untied, and learn more flexible(It is a neural network of just one layer.)|

###11. Cost function and Entropy Diff Graph

| Index           | Function          | Comments                               |
|-----------------|:------------------|:---------------------------------------|
|Cost Fig.        |cost and entropy Graph|Cost graph use Left side axis. The horizontal axis is Epoch Count. And entrpy-difference(decode data from training data) of each digit image use Right side axis.|

###12. W Weight Range and Bias b plot Graph

| Index           | Function          | Comments                               |
|-----------------|:------------------|:---------------------------------------|
|Last W Range Fig.|W Range of all Nodes|Red Error bar is W Weight Graph, and the axis is left side.Horizontal axis is Node Number.|
|last b Bios Fig. |b value of all Nodes|Blue dot is Bias b plot, and the axis is right side.|

###13. Encode value of selected MNIST sample (from 0 to 9 and Summary)
| Index           | Function          | Comments                               |
|-----------------|:------------------|:---------------------------------------|
|Last Z=f(Wx+b) Range Fig.|encode value of sample data and sum of them by each node|It was plotted by shifting the axis of the vertical.|

###14. Footer Information
| Index           | Function          | Comments                               |
|-----------------|:------------------|:---------------------------------------|
|Total Training time(sec)|the time from training start to finish learning|It does not have time are included to create a report.|
|W Entropy gain   |total entropy gain form start|same as the last Term Wgain|
|Sparse Simple Ratio(%)|How much sparse|(total square root of sum of abs(f(Wx+b) / max of f(Wx+b)) / count of W) * 100|
|Last Entropy Diff|Entropy diff of after decode from before encode|same as the last Term dx-ent.|
|W Region         |How meny features are localized|In somehow feeling|
|W Var            |variance of W Weight|varience of Last Hidden Layer(W Weight)|
|Img Diff         |the difference between Original and Decode|same as the last Term Img-diff|

## Licence

[MIT](https://github.com/tcnksm/tool/blob/master/LICENCE)

## Author

[np2lkoo](https://github.com/np2lkoo)
