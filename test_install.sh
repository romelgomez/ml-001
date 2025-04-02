#!/bin/bash

echo "Validating Hugging Face Transformers installation..."

# Run the sentiment analysis test
RESULT=$(python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('hugging face is the best'))")

echo "Test result: $RESULT"

# Check if the result contains "POSITIVE"
if [[ $RESULT == *"'label': 'POSITIVE'"* ]]; then
    echo "✅ Test passed! Transformers is correctly installed and working."
    echo "Label detected as POSITIVE as expected."
else
    echo "❌ Test failed. Expected POSITIVE sentiment but got something else."
    echo "Please check your installation of transformers."
fi

# Print additional info
echo ""
echo "Transformers version:"
python -c "import transformers; print(transformers.__version__)"

echo ""
echo "Dependencies:"
python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || echo "PyTorch: Not installed"
python -c "import tensorflow; print(f'TensorFlow: {tensorflow.__version__}')" 2>/dev/null || echo "TensorFlow: Not installed"