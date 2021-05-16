# HandDetectionSafetyProject
HAND DETECTION & SAFETY PROJECT
By :- SARANG TAMRAKAR

Problem Statement : - There is one sheder machine (which generally takes the Garbage as input) ,input provided by the worker of that company but while giving the input to machine there are high chances of Permanent hand damage due to rigid Jaw of sheder machine.
So our work to detect the hand of worker while passing the garbage to the machine  , there are two safety lines. 1. Blue line = for threatening the worker by ALERT sound 
                       2. Red Line= for stopping the machine if hand crosses to that red line & after hand
                                              returning start the machine again (by connecting the relay)

Solution:- This solution of this problem have to be in REAL TIME , so we have selected the SSD MOBILENET Model, which will detect the object into real time.

1.	DATA COLLECTION STRETEGY :- we have gone to the deployment area, then took the video from different angles & in different lightening conditions.

2.	Data preparation: - We have convert the video into the images then we have done data annotation through the Labelimg tool. (5000 images)


3.	Model Building:- we have trained the model in TFOD framework till 50000 EPOCHS.

4.	Inferencing :- we have taken the trained model & build the solution such that we are detecting hand if they are crossing lines so we are taking the actions according to that.

THANK YOU……
