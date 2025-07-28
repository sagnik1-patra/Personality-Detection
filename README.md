This project uses a deep learning model to predict a user's MBTI (Myers-Briggs Type Indicator) personality type based on text input.

 Project Structure
bash
Copy
Edit
New folder/
├── mbti_1.csv                   # Dataset (MBTI posts & labels)
├── train_personality_model.py  # Script to train model and save .h5 & tokenizer
├── personality_estimator.h5    # Trained model (Bidirectional LSTM)
├── tokenizer.joblib            # Tokenizer used for inference
├── predict_personality.py      # Run this script to predict MBTI from user input
└── README.md                   # Project documentation
 Requirements
Install dependencies using:

bash
Copy
Edit
pip install tensorflow numpy pandas scikit-learn joblib
🧪 Training the Model
To train the model on the MBTI dataset:

bash
Copy
Edit
python train_personality_model.py
 This will:

Clean and preprocess the dataset (mbti_1.csv)

Tokenize the text and encode MBTI labels

Train a Bidirectional LSTM model

Save the model as personality_estimator.h5

Save tokenizer as tokenizer.joblib

 Predicting Personality
To start predicting MBTI types from user input:

bash
Copy
Edit
python predict_personality.py
You’ll see:

pgsql
Copy
Edit
 Enter text for personality prediction (or type 'exit' to quit):
> I enjoy solving logic puzzles and thinking critically about problems.
 Predicted MBTI Type: INTJ (Confidence: 0.78)
 Model Details
Architecture: Embedding → Bidirectional LSTM → Dense → Softmax

Classes: ['INTP', 'INTJ', 'INFP', ..., 'ESFJ'] (16 MBTI types)

Tokenizer: Saved using joblib and used for inference

Loss & Metrics: categorical_crossentropy, accuracy

 Improvements Included
Class balancing using class_weight

Bidirectional LSTM for richer context

Confidence score from softmax output
