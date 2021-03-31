<h1> Predicting-Long-term-Student-Engagement-Cognitive-Psychology-in-Deep-Knowledge-Tracing </h1>

<h2> Abstract </h2>
  
  Knowledge tracing models predict learners’ correctness of answersto specific questions based on their prior performance. Too muchfocus has been placed on the individual question/responses andquestion recommendation systems ignore the more effective studyhabits. We introduce “spacing,” i.e., distributing study/practice overmore sessions, to the knowledge tracing domain. Spacing is anexperimentally proven cognitive psychology method used to im-prove long-term learning. Our model predicts the extent to whichstudents will use spacing in their studying. Instructors can thenidentify students who may engage in less effective learning sched-ules and motivate them while they still have sufficient time to doso. We implemented two models: 1) an LSTM-based model, wherethe final dense layer leverages summation of LSTM outputs, and 2)a transformer-based architecture with stacked encoders, composedof self-attention layers and feed-forward networks. We trained andtested these two models on two real-world datasets, a psychol-ogy MOOC dataset that contains students-module interactions onCoursera, and the large-scale EdNet dataset collected from an activetutoring application, SANTA. The LSTM model outperformed thetransformer model on the EdNet dataset, and feeding more daysof students’ learning activities to the model improved its perfor-mance monotonically. However, we did not observe any significantdifference between the two models on the MOOC data, which canbe explained by its small sample size.
  
  
  
 <h2> The Repository </h2> 
  This repository includes implementation of two models on the problem, an LSTM-based model, where the final dense layer leverages summation of LSTM outputs, and a transformer-based architecture with stacked encoders, composed of self-attention layers and feed-forward networks. The models were trained on EdNet dataset.
