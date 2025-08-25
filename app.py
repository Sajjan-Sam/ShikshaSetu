import streamlit as st
import json
import os
import uuid
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
import random
import re


def _get_api_key() -> str:
    """Get OpenAI API key from environment or Streamlit secrets"""
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        try:
            key = st.secrets.get("OPENAI_API_KEY", "").strip()
        except Exception:
            key = ""
    return key


def _letter_to_text(letter: str, options: list[str]) -> str | None:
    """Convert letter choice to option text"""
    if not isinstance(letter, str) or not options or len(options) < 4:
        return None
    letter = letter.strip().upper()
    if letter in "ABCD":
        return options["ABCD".index(letter)]
    if letter in options:
        return letter
    return None


def _extract_json_array(block: str):
    """Extract JSON array from text block"""
    import json, re
    m = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", block)
    s = m.group(1) if m else block
    m2 = re.search(r"(\[[\s\S]*\])", s)
    s = m2.group(1) if m2 else s
    return json.loads(s)


def _mcq_valid(stem: str, options: list[str], correct_text: str | None) -> bool:
    """Validate MCQ structure and content"""
    if not stem or not options or len(options) != 4: 
        return False
    if len(set(o.strip() for o in options)) != 4: 
        return False
    banned = {"all of the above", "none of the above", "both a and b", "all of these"}
    if any(o.strip().lower() in banned for o in options): 
        return False
    return isinstance(correct_text, str) and correct_text in options


# Page configuration
st.set_page_config(
    page_title="ShikshaSetu - Assignment Generator & Evaluator", 
    page_icon="🌉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling (same as before)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    .hero-container {
        background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);
        padding: 2rem;
        border-radius: 1rem;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        font-weight: 400;
        opacity: 0.9;
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.75rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid #f3f4f6;
        margin-bottom: 1rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    }
    
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .feature-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #374151;
        margin-bottom: 0.5rem;
    }
    
    .feature-description {
        color: #6b7280;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    .mcq-container {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .mcq-question {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 1rem;
    }
    
    .mcq-option {
        background: white;
        border: 1px solid #cbd5e1;
        border-radius: 0.375rem;
        padding: 0.75rem;
        margin: 0.5rem 0;
        display: flex;
        align-items: center;
    }
    
    .mcq-option.correct {
        background: #dcfce7;
        border-color: #16a34a;
        color: #166534;
    }
    
    .option-letter {
        font-weight: 700;
        margin-right: 0.75rem;
        min-width: 2rem;
    }
    
    .login-card {
        background: white;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
    }
    
    .login-header {
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    .login-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #374151;
        margin-bottom: 0.5rem;
    }
    
    .login-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        border: 1px solid #f3f4f6;
        text-align: center;
    }
    
    .custom-alert {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    
    .alert-info {
        background-color: #eff6ff;
        border-color: #3b82f6;
        color: #1e40af;
    }
    
    .alert-success {
        background-color: #f0fdf4;
        border-color: #22c55e;
        color: #166534;
    }
    
    .alert-warning {
        background-color: #fffbeb;
        border-color: #f59e0b;
        color: #92400e;
    }
    
    .alert-error {
        background-color: #fef2f2;
        border-color: #ef4444;
        color: #dc2626;
    }
</style>
""", unsafe_allow_html=True)

# Ensure data directory exists
os.makedirs("data", exist_ok=True)


def get_username_from_email(email: str) -> str:
    """Extract username from email address"""
    return email.split('@')[0]


def load_json(filepath: str) -> Dict[str, Any]:
    """Load JSON data from file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}


def save_json(data: Dict[str, Any], filepath: str) -> bool:
    """Save data to JSON file"""
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return False


def check_api_key_status() -> tuple[bool, str]:
    """Check if API key is available and provide status message"""
    key = _get_api_key()
    if not key:
        return False, "⚠️ OpenAI API key not found in secrets. Using fallback question generation."
    
    # Test the API key with a minimal request
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        
        # Make a minimal test request
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5,
            temperature=0
        )
        return True, "✅ OpenAI API key is working properly."
        
    except Exception as e:
        error_msg = str(e).lower()
        if "quota" in error_msg or "billing" in error_msg:
            return False, "💳 OpenAI API quota exceeded. Please check your billing and usage limits. Using fallback questions."
        elif "invalid" in error_msg or "authentication" in error_msg:
            return False, "🔑 Invalid OpenAI API key. Please check your key in secrets.toml. Using fallback questions."
        else:
            return False, f"❌ OpenAI API error: {str(e)[:100]}... Using fallback questions."


# Enhanced AI Agent 1: MCQ Generator with Better Error Handling
def generate_mcq_with_ai(topic: str, level: str, difficulty: int, count: int) -> List[Dict[str, Any]]:
    """
    Enhanced AI Agent 1: MCQ Generator with robust error handling
    """
    # Check API key status first
    api_available, status_msg = check_api_key_status()
    
    if not api_available:
        st.warning(status_msg)
        return generate_enhanced_fallback_mcqs(topic, level, count)
    
    difficulty_level = "introductory" if difficulty < 40 else "intermediate" if difficulty < 70 else "advanced"
    
    university_context = {
        ("UG", "introductory"): "MIT, Stanford, Harvard undergraduate introductory level",
        ("UG", "intermediate"): "MIT, Stanford, Harvard, Oxford, Cambridge undergraduate intermediate level",
        ("UG", "advanced"): "MIT, Stanford, Harvard, Oxford, Cambridge undergraduate advanced level",
        ("PG", "introductory"): "MIT, Stanford, Harvard, Oxford, Cambridge graduate introductory level",
        ("PG", "intermediate"): "MIT, Stanford, Harvard, Oxford, Cambridge graduate intermediate level", 
        ("PG", "advanced"): "MIT, Stanford, Harvard, Oxford, Cambridge graduate advanced research level"
    }[(level, difficulty_level)]

    prompt = f"""
You are an elite exam question creator from top universities like MIT, Stanford, Harvard, Oxford, and Cambridge.

Generate {count} rigorous, high-quality multiple-choice questions on "{topic}" at {university_context}.

STRICT REQUIREMENTS:
1. Each question must have EXACTLY 4 unique options (A, B, C, D)
2. Only ONE option is correct
3. No "All of the above", "None of the above", or similar meta-options
4. Questions should be challenging and intellectually stimulating
5. Options should be plausible but clearly distinguishable
6. Include comprehensive explanations for the correct answer

Return ONLY a valid JSON array with this exact schema:
[
  {{
    "question": "Clear, precise question text",
    "options": ["Option A text", "Option B text", "Option C text", "Option D text"],
    "correct_index": 0,
    "explanation": "Detailed explanation why this answer is correct and others are wrong"
  }}
]

Topic: {topic}
Level: {level}
Difficulty: {difficulty_level}
Count: {count}
"""

    try:
        from openai import OpenAI
        key = _get_api_key()
        client = OpenAI(api_key=key)

        with st.spinner("🤖 AI is creating high-quality questions..."):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.3,
                max_tokens=4000,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert academic question creator from elite universities. You produce rigorous, high-quality MCQs and return ONLY valid JSON arrays."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ]
            )

        data = _extract_json_array(response.choices[0].message.content)

        # Validate and clean the generated questions
        validated_questions = []
        for item in data:
            try:
                question_text = (item.get("question") or "").strip()
                options = item.get("options") or []
                correct_idx = item.get("correct_index")
                explanation = (item.get("explanation") or "").strip()

                # Validate structure
                if not (isinstance(correct_idx, int) and 0 <= correct_idx < 4):
                    continue
                if not options or len(options) != 4:
                    continue
                if not question_text or not explanation:
                    continue

                correct_text = options[correct_idx]
                
                # Validate content quality
                if not _mcq_valid(question_text, options, correct_text):
                    continue

                # Create standardized format
                validated_questions.append({
                    "question": question_text,
                    "options": {
                        "A": options[0],
                        "B": options[1], 
                        "C": options[2],
                        "D": options[3]
                    },
                    "correct_answer": "ABCD"[correct_idx],
                    "explanation": explanation
                })

            except Exception:
                continue  # Skip invalid questions

        if validated_questions:
            st.success(f"✅ AI generated {len(validated_questions)} high-quality questions!")
            return validated_questions
        else:
            st.warning("⚠️ AI generated questions didn't meet quality standards. Using enhanced fallback.")
            return generate_enhanced_fallback_mcqs(topic, level, count)

    except Exception as e:
        error_msg = str(e)
        if "quota" in error_msg.lower() or "billing" in error_msg.lower():
            st.error("💳 OpenAI API quota exceeded. Please check your billing and usage limits.")
        elif "invalid" in error_msg.lower() or "authentication" in error_msg.lower():
            st.error("🔑 Invalid OpenAI API key. Please check your key configuration.")
        else:
            st.error(f"❌ AI generation failed: {error_msg}")
        
        st.info("🔄 Using high-quality fallback questions instead.")
        return generate_enhanced_fallback_mcqs(topic, level, count)


def generate_enhanced_fallback_mcqs(topic: str, level: str, count: int) -> List[Dict[str, Any]]:
    """Generate comprehensive high-quality fallback MCQs for various topics"""
    
    # Expanded question bank with more topics and difficulty levels
    question_bank = {
        "Computer Science": {
            "UG": [
                {
                    "question": "What is the primary advantage of using dynamic programming in algorithm design?",
                    "options": {
                        "A": "It reduces space complexity to O(1) for all problems",
                        "B": "It eliminates the need for recursion entirely", 
                        "C": "It avoids redundant calculations by storing intermediate results",
                        "D": "It guarantees finding the optimal solution in constant time"
                    },
                    "correct_answer": "C",
                    "explanation": "Dynamic programming's main advantage is avoiding redundant calculations by storing and reusing intermediate results (memoization), which can dramatically improve time complexity from exponential to polynomial in many cases."
                },
                {
                    "question": "In object-oriented programming, what is the main purpose of encapsulation?",
                    "options": {
                        "A": "To increase program execution speed significantly",
                        "B": "To hide internal implementation details and control access to data",
                        "C": "To enable multiple inheritance from different classes",
                        "D": "To automatically optimize memory usage"
                    },
                    "correct_answer": "B", 
                    "explanation": "Encapsulation primarily serves to hide internal implementation details and control access to object properties and methods, promoting data integrity and code maintainability."
                },
                {
                    "question": "Which data structure is most efficient for implementing a priority queue?",
                    "options": {
                        "A": "Linked List with sorted insertion",
                        "B": "Binary Search Tree with balancing",
                        "C": "Binary Heap (Min-heap or Max-heap)",
                        "D": "Hash Table with priority indexing"
                    },
                    "correct_answer": "C",
                    "explanation": "Binary heaps provide O(log n) insertion and deletion of the highest priority element, making them the most efficient standard implementation for priority queues."
                },
                {
                    "question": "What is the time complexity of the quicksort algorithm in the average case?",
                    "options": {
                        "A": "O(n)",
                        "B": "O(n log n)",
                        "C": "O(n²)",
                        "D": "O(log n)"
                    },
                    "correct_answer": "B",
                    "explanation": "Quicksort has an average-case time complexity of O(n log n) when the pivot divides the array reasonably evenly. The worst case is O(n²) when the pivot is always the smallest or largest element."
                },
                {
                    "question": "In database design, what is the primary goal of normalization?",
                    "options": {
                        "A": "To increase query execution speed",
                        "B": "To reduce data redundancy and prevent anomalies",
                        "C": "To maximize storage space utilization",
                        "D": "To simplify database backup procedures"
                    },
                    "correct_answer": "B",
                    "explanation": "Database normalization primarily aims to reduce data redundancy and prevent update, insertion, and deletion anomalies by organizing data into well-structured tables."
                }
            ],
            "PG": [
                {
                    "question": "In distributed systems, what is the fundamental trade-off described by the CAP theorem?",
                    "options": {
                        "A": "Between computational speed, memory usage, and network bandwidth",
                        "B": "Between Consistency, Availability, and Partition tolerance",
                        "C": "Between security, scalability, and maintainability",
                        "D": "Between latency, throughput, and fault tolerance"
                    },
                    "correct_answer": "B",
                    "explanation": "The CAP theorem states that distributed systems can guarantee at most two out of three properties: Consistency (all nodes see the same data simultaneously), Availability (system remains operational), and Partition tolerance (system continues despite network failures)."
                },
                {
                    "question": "Which machine learning technique is most suitable for handling sequential data with long-term dependencies?",
                    "options": {
                        "A": "Convolutional Neural Networks (CNNs)",
                        "B": "Support Vector Machines (SVMs)",
                        "C": "Long Short-Term Memory networks (LSTMs)",
                        "D": "Random Forest algorithms"
                    },
                    "correct_answer": "C",
                    "explanation": "LSTMs are specifically designed to handle sequential data and can capture long-term dependencies through their gating mechanisms, making them ideal for time series, natural language processing, and other sequential tasks."
                }
            ]
        },
        "Computer Vision": {
            "UG": [
                {
                    "question": "What is the primary purpose of convolutional layers in a CNN?",
                    "options": {
                        "A": "To reduce the dimensionality of input images permanently",
                        "B": "To extract local spatial features through learnable filters", 
                        "C": "To perform the final classification of image content",
                        "D": "To normalize pixel values across color channels"
                    },
                    "correct_answer": "B",
                    "explanation": "Convolutional layers use learnable filters (kernels) to extract local spatial features like edges, textures, and patterns from input images through convolution operations, preserving spatial relationships."
                },
                {
                    "question": "Which technique is most effective for preventing overfitting in deep neural networks?",
                    "options": {
                        "A": "Increasing the learning rate significantly",
                        "B": "Adding more hidden layers to the network",
                        "C": "Applying dropout and regularization techniques",
                        "D": "Using smaller batch sizes during training"
                    },
                    "correct_answer": "C",
                    "explanation": "Dropout randomly sets some neurons to zero during training, and regularization techniques (L1/L2) add penalty terms to prevent overfitting by reducing model complexity and improving generalization."
                }
            ],
            "PG": [
                {
                    "question": "In advanced computer vision, what is the key innovation of attention mechanisms?",
                    "options": {
                        "A": "They eliminate the need for convolutional operations entirely",
                        "B": "They allow models to focus on relevant parts of input dynamically",
                        "C": "They automatically augment training data for better performance",
                        "D": "They reduce computational requirements to linear complexity"
                    },
                    "correct_answer": "B",
                    "explanation": "Attention mechanisms allow models to dynamically focus on the most relevant parts of the input, enabling better handling of long sequences and complex spatial relationships in vision tasks."
                }
            ]
        },
        "Machine Learning": {
            "UG": [
                {
                    "question": "What is the bias-variance tradeoff in machine learning?",
                    "options": {
                        "A": "The balance between training speed and prediction accuracy",
                        "B": "The tradeoff between model complexity and interpretability",
                        "C": "The balance between underfitting and overfitting tendencies",
                        "D": "The choice between supervised and unsupervised learning"
                    },
                    "correct_answer": "C",
                    "explanation": "The bias-variance tradeoff refers to the balance between bias (underfitting - too simple models) and variance (overfitting - too complex models) to achieve optimal generalization performance."
                }
            ],
            "PG": [
                {
                    "question": "In reinforcement learning, what does the exploration-exploitation dilemma address?",
                    "options": {
                        "A": "Whether to use model-based or model-free approaches",
                        "B": "The balance between trying new actions vs. using known good actions",
                        "C": "The choice between on-policy and off-policy methods",
                        "D": "Whether to use discrete or continuous action spaces"
                    },
                    "correct_answer": "B",
                    "explanation": "The exploration-exploitation dilemma is the fundamental challenge of balancing the exploration of new actions (to discover potentially better strategies) with exploitation of currently known good actions (to maximize immediate rewards)."
                }
            ]
        },
        "Mathematics": {
            "UG": [
                {
                    "question": "What is the geometric interpretation of the determinant of a 2×2 matrix?",
                    "options": {
                        "A": "The length of the diagonal vector",
                        "B": "The area of the parallelogram formed by column vectors",
                        "C": "The angle between the two column vectors",
                        "D": "The sum of all matrix elements"
                    },
                    "correct_answer": "B",
                    "explanation": "The determinant of a 2×2 matrix represents the signed area of the parallelogram formed by its column vectors, with the sign indicating orientation."
                }
            ],
            "PG": [
                {
                    "question": "In functional analysis, what defines a Hilbert space?",
                    "options": {
                        "A": "A normed space that is complete under its norm",
                        "B": "A complete inner product space",
                        "C": "A finite-dimensional vector space with orthogonal basis",
                        "D": "A metric space with translation invariance"
                    },
                    "correct_answer": "B",
                    "explanation": "A Hilbert space is a complete inner product space, meaning it has an inner product that induces a norm, and every Cauchy sequence converges within the space."
                }
            ]
        }
    }
    
    # Get appropriate questions for the topic and level
    topic_questions = question_bank.get(topic, question_bank["Computer Science"])
    level_questions = topic_questions.get(level, topic_questions.get("UG", []))
    
    # If we don't have enough predefined questions, generate generic ones
    fallback_questions = []
    
    # Use available predefined questions first
    available_questions = min(len(level_questions), count)
    fallback_questions.extend(level_questions[:available_questions])
    
    # Generate additional generic questions if needed
    remaining_count = count - available_questions
    for i in range(remaining_count):
        question_num = available_questions + i + 1
        fallback_questions.append({
            "question": f"Which of the following best describes a fundamental concept in {topic} at the {level} level?",
            "options": {
                "A": f"Theoretical foundations and mathematical models in {topic}",
                "B": f"Practical applications and real-world implementations of {topic}",
                "C": f"Historical development and evolution of {topic} methodologies", 
                "D": f"Interdisciplinary connections between {topic} and related fields"
            },
            "correct_answer": "A",
            "explanation": f"Theoretical foundations and mathematical models provide the core conceptual framework necessary for understanding {topic} at the {level} level, forming the basis for all practical applications and advanced study."
        })
    
    return fallback_questions


# Enhanced AI Agent 2: Answer Evaluator with Better Error Handling
def evaluate_answer_with_ai(
    question: str,
    student_answer: str,
    correct_answer: str,
    question_type: str,
    options: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Enhanced AI Agent 2: Answer Evaluator with robust error handling
    """
    
    # MCQ Evaluation (Deterministic)
    if question_type == "mcq":
        if options:
            options_list = [options.get("A"), options.get("B"), options.get("C"), options.get("D")]
            correct_text = _letter_to_text(correct_answer, options_list)
            if not correct_text:
                correct_text = correct_answer
        else:
            correct_text = correct_answer

        is_correct = (student_answer or "").strip() == (correct_text or "").strip()
        
        if is_correct:
            feedback = "✅ Excellent! Your answer is correct."
        else:
            feedback = f"❌ Incorrect. The correct answer is: {correct_text}"
            
        return {
            "score": 1 if is_correct else 0,
            "max_score": 1,
            "feedback": feedback,
            "percentage": 100 if is_correct else 0,
        }

    # Subjective Answer Evaluation with better error handling
    api_available, _ = check_api_key_status()
    
    if not api_available:
        # Enhanced fallback evaluation for subjective questions
        return evaluate_subjective_fallback(question, student_answer, correct_answer)
    
    try:
        from openai import OpenAI
        key = _get_api_key()
        client = OpenAI(api_key=key)
        
        evaluation_prompt = f"""
You are a university professor evaluating student answers with the rigor of top institutions.

Evaluate this response fairly and constructively:

QUESTION: {question}

EXPECTED/REFERENCE ANSWER: {correct_answer}

STUDENT'S ANSWER: {student_answer}

EVALUATION CRITERIA:
1. Correctness and accuracy (40%)
2. Depth of understanding (25%)
3. Clarity and organization (20%)
4. Use of examples/evidence (15%)

Provide a score out of 10 and detailed feedback (2-4 sentences).

Return ONLY this JSON format:
{{
    "score": <integer 0-10>,
    "feedback": "Detailed constructive feedback explaining the score"
}}
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {
                    "role": "system", 
                    "content": "You are a fair university professor providing detailed academic evaluations. Always return valid JSON only."
                },
                {
                    "role": "user", 
                    "content": evaluation_prompt
                }
            ]
        )

        # Extract JSON from response
        content = response.choices[0].message.content
        json_match = re.search(r"(\{[\s\S]*\})", content)
        
        if json_match:
            result = json.loads(json_match.group(1))
        else:
            result = json.loads(content)

        score = max(0, min(int(result.get("score", 0)), 10))
        feedback = (result.get("feedback") or "").strip() or "Answer evaluated."
        percentage = round(score * 10, 1)

        return {
            "score": score,
            "max_score": 10,
            "feedback": feedback,
            "percentage": percentage
        }

    except Exception as e:
        st.warning(f"AI evaluation failed: {str(e)}. Using fallback evaluation.")
        return evaluate_subjective_fallback(question, student_answer, correct_answer)


def evaluate_subjective_fallback(question: str, student_answer: str, correct_answer: str) -> Dict[str, Any]:
    """Enhanced fallback evaluation for subjective questions"""
    
    if not student_answer or not student_answer.strip():
        return {
            "score": 0,
            "max_score": 10,
            "feedback": "❌ No answer provided. Please provide a response to receive credit.",
            "percentage": 0
        }
    
    answer_length = len(student_answer.strip().split())
    answer_lower = student_answer.lower()
    
    # Basic content analysis
    score = 2  # Base score for attempting the question
    
    # Length-based scoring
    if answer_length >= 50:
        score += 2
    elif answer_length >= 20:
        score += 1
    
    # Keyword matching with expected answer
    expected_words = set(correct_answer.lower().split())
    student_words = set(answer_lower.split())
    
    # Calculate overlap with expected answer
    common_words = expected_words.intersection(student_words)
    if expected_words:
        overlap_ratio = len(common_words) / len(expected_words)
        score += min(3, int(overlap_ratio * 6))  # Up to 3 additional points
    
    # Structure and effort indicators
    if any(word in answer_lower for word in ['example', 'instance', 'such as', 'for example']):
        score += 1
    
    if any(word in answer_lower for word in ['because', 'therefore', 'thus', 'hence', 'consequently']):
        score += 1
    
    if any(word in answer_lower for word in ['however', 'although', 'nevertheless', 'on the other hand']):
        score += 1
    
    score = min(score, 10)  # Cap at maximum score
    
    # Generate feedback based on score
    if score >= 8:
        feedback = "✅ Excellent response! Your answer demonstrates strong understanding with good examples and reasoning."
    elif score >= 6:
        feedback = "👍 Good answer! Your response shows understanding but could benefit from more detail or examples."
    elif score >= 4:
        feedback = "📚 Fair attempt. Your answer has some relevant points but needs more depth and clarity."
    else:
        feedback = "⚠️ Basic response. Please provide more detailed explanations and specific examples to improve your score."
    
    return {
        "score": score,
        "max_score": 10,
        "feedback": feedback,
        "percentage": round(score * 10, 1)
    }


def generate_items(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate assignment items with enhanced AI-powered MCQs"""
    items = []
    item_counter = 1
    
    # Generate AI-powered MCQ items
    if config.get('mcq_count', 0) > 0:
        ai_mcqs = generate_mcq_with_ai(
            config['topic'], 
            config['level'], 
            config['difficulty'], 
            config['mcq_count']
        )
        
        for mcq in ai_mcqs:
            items.append({
                "id": f"MCQ-{item_counter:03d}",
                "type": "mcq",
                "stem": mcq["question"],
                "options": mcq["options"],
                "answer_key": mcq["correct_answer"],
                "explanation": mcq.get("explanation", ""),
                "rubric": []
            })
            item_counter += 1
    
    # Generate Short Answer items
    for i in range(config.get('short_count', 0)):
        items.append({
            "id": f"SA-{item_counter:03d}",
            "type": "short",
            "stem": f"Explain a key concept in {config['topic']} relevant to {config['level']} level understanding. Provide specific examples and discuss its practical implications.",
            "options": {},
            "answer_key": f"A comprehensive answer should include: definition of the key concept, theoretical background, practical examples from {config['topic']}, and discussion of real-world applications.",
            "rubric": [
                {
                    "criterion": "Content Accuracy",
                    "levels": [
                        {"score": 5, "description": "Completely accurate with excellent understanding"},
                        {"score": 4, "description": "Mostly accurate with good understanding"},
                        {"score": 3, "description": "Generally accurate with adequate understanding"},
                        {"score": 2, "description": "Partially accurate with basic understanding"},
                        {"score": 1, "description": "Limited accuracy with poor understanding"},
                        {"score": 0, "description": "Inaccurate or no understanding demonstrated"}
                    ]
                }
            ]
        })
        item_counter += 1
    
    # Generate Long Answer items  
    for i in range(config.get('long_count', 0)):
        items.append({
            "id": f"LA-{item_counter:03d}",
            "type": "long", 
            "stem": f"Critically analyze and evaluate the role of {config['topic']} in modern applications. Discuss theoretical foundations, current challenges, and future prospects with supporting evidence.",
            "options": {},
            "answer_key": f"A comprehensive response should cover: theoretical foundations of {config['topic']}, current applications and case studies, analysis of challenges and limitations, future trends and developments, critical evaluation with supporting evidence.",
            "rubric": [
                {
                    "criterion": "Content Knowledge & Analysis",
                    "levels": [
                        {"score": 15, "description": "Exceptional depth and critical analysis"},
                        {"score": 12, "description": "Good depth with solid analysis"}, 
                        {"score": 9, "description": "Adequate depth with basic analysis"},
                        {"score": 6, "description": "Limited depth with minimal analysis"},
                        {"score": 3, "description": "Poor depth with no real analysis"},
                        {"score": 0, "description": "No meaningful content or analysis"}
                    ]
                }
            ]
        })
        item_counter += 1
    
    return items


def display_mcq_for_professor(question_data: Dict[str, Any]) -> None:
    """Display MCQ in a clean, professional format for professors"""
    
    question_id = question_data.get('id', 'Unknown')
    question_text = question_data.get('stem', '')
    options = question_data.get('options', {})
    correct_answer = question_data.get('answer_key', '')
    explanation = question_data.get('explanation', '')
    
    # Handle both dict and list formats for options
    if isinstance(options, list):
        if len(options) >= 4:
            options_dict = {
                'A': options[0],
                'B': options[1],
                'C': options[2], 
                'D': options[3]
            }
        else:
            options_dict = {'A': '', 'B': '', 'C': '', 'D': ''}
    elif isinstance(options, dict):
        options_dict = options
    else:
        options_dict = {'A': '', 'B': '', 'C': '', 'D': ''}
    
    # Create the MCQ container
    st.markdown(f"""
    <div class="mcq-container">
        <div class="mcq-question">
            <strong>{question_id}:</strong> {question_text}
        </div>
    """, unsafe_allow_html=True)
    
    # Display options
    for letter in ['A', 'B', 'C', 'D']:
        option_text = options_dict.get(letter, '')
        if option_text.strip():  # Only display non-empty options
            is_correct = (letter == correct_answer)
            
            option_class = "mcq-option correct" if is_correct else "mcq-option"
            correct_indicator = " ✅" if is_correct else ""
            
            st.markdown(f"""
            <div class="{option_class}">
                <span class="option-letter">{letter}.</span>
                <span>{option_text}{correct_indicator}</span>
            </div>
            """, unsafe_allow_html=True)
    
    # Display explanation if available
    if explanation and explanation.strip():
        st.markdown(f"""
        <div style="margin-top: 1rem; padding: 1rem; background: #f1f5f9; border-radius: 0.375rem; border-left: 4px solid #3b82f6;">
            <strong>💡 Explanation:</strong> {explanation}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)


# Authentication functions
def is_valid_email(email: str) -> bool:
    """Validate email format"""
    gmail_pattern = r'^[a-zA-Z0-9._%+-]+@gmail\.com$'
    college_pattern = r'^[a-zA-Z0-9._%+-]+@iiserb\.ac\.in$'
    return re.match(gmail_pattern, email) is not None or re.match(college_pattern, email) is not None


def load_users() -> Dict[str, Any]:
    """Load user data"""
    return load_json("data/users.json")


def save_users(users: Dict[str, Any]) -> bool:
    """Save user data"""
    return save_json(users, "data/users.json")


def authenticate_user(email: str, role: str) -> bool:
    """Authenticate user and check if they're allowed"""
    users = load_users()
    allowed_users = users.get("allowed_users", {})
    
    if email in allowed_users:
        return allowed_users[email]["role"] == role
    return False


def add_allowed_user(email: str, role: str) -> bool:
    """Add user to allowed list (professor only)"""
    users = load_users()
    if "allowed_users" not in users:
        users["allowed_users"] = {}
    
    users["allowed_users"][email] = {
        "role": role,
        "added_at": datetime.now().isoformat()
    }
    return save_users(users)


def delete_student(email: str) -> bool:
    """Delete student from allowed list (professor only)"""
    users = load_users()
    allowed_users = users.get("allowed_users", {})
    
    if email in allowed_users and allowed_users[email]["role"] == "student":
        del allowed_users[email]
        users["allowed_users"] = allowed_users
        return save_users(users)
    return False


def get_published_assignments() -> List[Dict[str, Any]]:
    """Get list of published assignments"""
    assignments_data = load_json("data/published_assignments.json")
    return assignments_data.get("assignments", [])


def publish_assignment(assignment: Dict[str, Any]) -> bool:
    """Publish assignment for students"""
    assignments_data = load_json("data/published_assignments.json")
    if "assignments" not in assignments_data:
        assignments_data["assignments"] = []
    
    # Remove existing assignment with same ID
    assignments_data["assignments"] = [a for a in assignments_data["assignments"] if a["id"] != assignment["id"]]
    
    # Add new assignment
    assignment["published_at"] = datetime.now().isoformat()
    assignment["scores_released"] = False
    assignments_data["assignments"].append(assignment)
    
    return save_json(assignments_data, "data/published_assignments.json")


def release_scores(assignment_id: str) -> bool:
    """Release scores for an assignment"""
    assignments_data = load_json("data/published_assignments.json")
    for assignment in assignments_data.get("assignments", []):
        if assignment["id"] == assignment_id:
            assignment["scores_released"] = True
            return save_json(assignments_data, "data/published_assignments.json")
    return False


def save_evaluation_results(student_email: str, assignment_id: str, results: Dict[str, Any]) -> bool:
    """Save evaluation results"""
    eval_data = {
        "student_email": student_email,
        "assignment_id": assignment_id,
        "results": results,
        "evaluated_at": datetime.now().isoformat()
    }
    return save_json(eval_data, f"data/evaluation_{student_email}_{assignment_id}.json")


def get_student_results(student_email: str) -> List[Dict[str, Any]]:
    """Get results for a specific student"""
    results = []
    assignments = get_published_assignments()
    
    for assignment in assignments:
        if assignment.get("scores_released", False):
            # Look for student's response file
            response_file = f"data/response_{student_email}_{assignment['id']}.json"
            if os.path.exists(response_file):
                response_data = load_json(response_file)
                
                # Look for evaluation results
                eval_file = f"data/evaluation_{student_email}_{assignment['id']}.json"
                if os.path.exists(eval_file):
                    eval_data = load_json(eval_file)
                    results.append({
                        "assignment": assignment,
                        "response": response_data,
                        "evaluation": eval_data
                    })
    
    return results


def delete_assignment(assignment_id: str) -> bool:
    """Delete a published assignment and all related data"""
    try:
        # Remove from published assignments
        assignments_data = load_json("data/published_assignments.json")
        if "assignments" in assignments_data:
            assignments_data["assignments"] = [a for a in assignments_data["assignments"] if a["id"] != assignment_id]
            save_json(assignments_data, "data/published_assignments.json")
        
        # Remove student responses
        response_files = [f for f in os.listdir("data") if f.startswith(f"response_") and f.endswith(f"_{assignment_id}.json")]
        for file in response_files:
            os.remove(f"data/{file}")
        
        # Remove evaluation results
        eval_files = [f for f in os.listdir("data") if f.endswith(f"_{assignment_id}.json") and f.startswith("evaluation_")]
        for file in eval_files:
            os.remove(f"data/{file}")
        
        return True
    except Exception as e:
        st.error(f"Error deleting assignment: {str(e)}")
        return False


# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_email' not in st.session_state:
    st.session_state.user_email = ""
if 'user_role' not in st.session_state:
    st.session_state.user_role = ""
if 'assignment' not in st.session_state:
    st.session_state.assignment = None
if 'responses' not in st.session_state:
    st.session_state.responses = {}


# Main Authentication Interface
if not st.session_state.authenticated:
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🌉 ShikshaSetu</div>
        <div class="hero-subtitle">A bridge from effort to mastery</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show API status
    api_available, status_msg = check_api_key_status()
    if api_available:
        st.markdown(f"""
        <div class="custom-alert alert-success">
            {status_msg}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="custom-alert alert-warning">
            {status_msg}
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🚀</div>
            <div class="feature-title">Smart Generation</div>
            <div class="feature-description">AI-powered assignment creation with customizable difficulty levels and multiple question types</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <div class="feature-title">Auto Evaluation</div>
            <div class="feature-description">Intelligent grading system with detailed rubrics and comprehensive feedback generation</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Progress Tracking</div>
            <div class="feature-description">Comprehensive performance analytics and detailed student progress monitoring</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 🔐 Access Your Account")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="login-card">
            <div class="login-header">
                <div class="login-icon">👨‍🏫</div>
                <div class="login-title">Professor Portal</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        prof_email = st.text_input("📧 Professor Email", placeholder="professor@gmail.com", key="prof_email")
        
        if st.button("🚀 Enter Professor Dashboard", type="primary", use_container_width=True):
            if is_valid_email(prof_email):
                st.session_state.authenticated = True
                st.session_state.user_email = prof_email
                st.session_state.user_role = "professor"
                st.rerun()
            else:
                st.error("⚠️ Please enter a valid email address")
    
    with col2:
        st.markdown("""
        <div class="login-card">
            <div class="login-header">
                <div class="login-icon">👨‍🎓</div>
                <div class="login-title">Student Portal</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        student_email = st.text_input("📧 Student Email", placeholder="student@iiserb.ac.in", key="student_email")
        
        if st.button("📚 Access Student Dashboard", type="primary", use_container_width=True):
            if is_valid_email(student_email):
                if authenticate_user(student_email, "student"):
                    st.session_state.authenticated = True
                    st.session_state.user_email = student_email
                    st.session_state.user_role = "student"
                    st.rerun()
                else:
                    st.error("🚫 You are not authorized to access this system. Please contact your professor.")
            else:
                st.error("⚠️ Please enter a valid email address")
    
    st.markdown("""
    <div class="custom-alert alert-info">
        <strong>🔧 Note for Students:</strong> You must be added by your professor before accessing the system. Contact your instructor if you're having trouble logging in.
    </div>
    """, unsafe_allow_html=True)
    
    st.stop()


# Main Application Dashboard
username = get_username_from_email(st.session_state.user_email)
st.markdown(f"""
<div class="hero-container">
    <div class="hero-title">🌉 ShikshaSetu Dashboard</div>
    <div class="hero-subtitle">Hello, {username}! ({st.session_state.user_role.title()})</div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([6, 1, 1])
with col3:
    if st.button("🚪 Logout", key="logout", help="Sign out of your account"):
        st.session_state.authenticated = False
        st.session_state.user_email = ""
        st.session_state.user_role = ""
        st.rerun()


# Professor Dashboard
if st.session_state.user_role == "professor":
    tab1, tab2, tab3, tab4 = st.tabs(["👥 Manage Students", "📝 Create Assignment", "📊 Evaluate & Release", "📈 Analytics Dashboard"])
    
    with tab1:
        st.markdown("### 👥 Student Management Center")
        
        with st.container():
            st.markdown("""
            <div class="feature-card">
                <div class="feature-title">➕ Add New Student</div>
                <div class="feature-description">Grant access to students by adding their email addresses</div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                new_student_email = st.text_input("📧 Student Email Address", placeholder="student@iiserb.ac.in")
            with col2:
                st.write("")  # Spacing
                if st.button("➕ Add Student", type="primary"):
                    if is_valid_email(new_student_email):
                        if add_allowed_user(new_student_email, "student"):
                            st.success(f"✅ Student {new_student_email} added successfully!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to add student")
                    else:
                        st.error("⚠️ Please enter a valid email address")
        
        st.markdown("### 📋 Current Students")
        users = load_users()
        students = {email: data for email, data in users.get("allowed_users", {}).items() if data["role"] == "student"}
        
        if students:
            for i, (email, data) in enumerate(students.items()):
                with st.container():
                    col1, col2, col3 = st.columns([4, 2, 1])
                    with col1:
                        st.markdown(f"**📧 {email}**")
                    with col2:
                        st.markdown(f"*Added: {data['added_at'][:10]}*")
                    with col3:
                        if st.button("🗑️", key=f"delete_{email}", help=f"Remove {email}", type="secondary"):
                            if delete_student(email):
                                st.success(f"✅ Student {email} removed successfully!")
                                st.rerun()
                            else:
                                st.error("❌ Failed to remove student")
                    st.divider()
        else:
            st.markdown("""
            <div class="custom-alert alert-info">
                <strong>📝 No students added yet</strong><br>
                Start by adding student email addresses above to grant them access to assignments.
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### 📝 Assignment Creation Studio")
        
        with st.sidebar:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-title">⚙️ Course Configuration</div>
                <div class="feature-description">Customize your assignment parameters</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show API status in sidebar
            api_available, status_msg = check_api_key_status()
            if api_available:
                st.success("🤖 AI Generation: Available")
            else:
                st.warning("🔧 AI Generation: Using Fallback Mode")
                with st.expander("ℹ️ Why am I seeing this?"):
                    st.info(status_msg)
                    st.markdown("""
                    **How to fix:**
                    1. Check your OpenAI API key in secrets.toml
                    2. Ensure you have sufficient credits/quota
                    3. Verify your key has the correct permissions
                    
                    **Don't worry:** The app will use high-quality fallback questions!
                    """)
            
            topic = st.text_input("📚 Topic", value="Computer Science", help="Enter the subject topic")
            level = st.selectbox("🎓 Academic Level", ["UG", "PG"], help="Select academic level")
            
            st.markdown("#### 📋 Question Types")
            mcq_enabled = st.checkbox("📘 Multiple Choice Questions", value=True)
            short_enabled = st.checkbox("✏️ Short Answer Questions", value=True)
            long_enabled = st.checkbox("📄 Long Answer Questions", value=False)
            
            st.markdown("#### 🔢 Question Counts")
            mcq_count = st.number_input("MCQ Count", min_value=0, max_value=20, value=5 if mcq_enabled else 0)
            short_count = st.number_input("Short Answer Count", min_value=0, max_value=10, value=3 if short_enabled else 0)
            long_count = st.number_input("Long Answer Count", min_value=0, max_value=5, value=2 if long_enabled else 0)
            
            difficulty = st.slider("🎯 Difficulty Level", min_value=0, max_value=100, value=50, help="Set difficulty level (0-100)")
            
            if st.button("🚀 Generate Assignment", type="primary", use_container_width=True):
                config = {
                    'topic': topic,
                    'level': level,
                    'mcq_count': mcq_count,
                    'short_count': short_count,
                    'long_count': long_count,
                    'difficulty': difficulty
                }
                
                items = generate_items(config)
                st.session_state.assignment = {
                    'id': str(uuid.uuid4()),
                    'config': config,
                    'items': items,
                    'created_at': datetime.now().isoformat(),
                    'created_by': st.session_state.user_email
                }
                
                st.success(f"✅ Generated {len(items)} questions!")
        
        if st.session_state.assignment is None:
            st.markdown("""
            <div class="feature-card" style="text-align: center; padding: 3rem;">
                <div class="feature-icon">📝</div>
                <div class="feature-title">Ready to Create?</div>
                <div class="feature-description">Configure your assignment settings in the sidebar and click 'Generate Assignment' to get started.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            assignment = st.session_state.assignment
            
            # Assignment summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div class="metric-container">
                    <div style="font-size: 1.5rem; font-weight: 600; color: #3b82f6;">{assignment['config']['topic']}</div>
                    <div style="color: #6b7280;">Topic</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-container">
                    <div style="font-size: 1.5rem; font-weight: 600; color: #3b82f6;">{assignment['config']['level']}</div>
                    <div style="color: #6b7280;">Level</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-container">
                    <div style="font-size: 1.5rem; font-weight: 600; color: #3b82f6;">{len(assignment['items'])}</div>
                    <div style="color: #6b7280;">Questions</div>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                <div class="metric-container">
                    <div style="font-size: 1.5rem; font-weight: 600; color: #3b82f6;">{assignment['config']['difficulty']}%</div>
                    <div style="color: #6b7280;">Difficulty</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Display MCQs in professional format
            mcq_items = [item for item in assignment["items"] if item.get("type") == "mcq"]
            if mcq_items:
                st.markdown("#### 🧩 Generated Multiple Choice Questions")
                for mcq_item in mcq_items:
                    display_mcq_for_professor(mcq_item)
            
            # Preview other question types
            other_items = [item for item in assignment["items"] if item.get("type") != "mcq"]
            if other_items:
                with st.expander("👀 Preview Other Questions", expanded=False):
                    for item in other_items:
                        st.markdown(f"**{item['id']}** ({item['type'].upper()})")
                        st.markdown(f"**Question:** {item['stem']}")
                        st.markdown(f"**Expected Answer:** {item['answer_key']}")
                        st.markdown("---")
            
            # Action buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🌐 Publish Assignment", type="primary", use_container_width=True):
                    if publish_assignment(st.session_state.assignment):
                        st.success("✅ Assignment published! Students can now access it.")
                        st.balloons()
            with col2:
                if st.button("💾 Save as Draft", use_container_width=True):
                    if save_json(st.session_state.assignment, "data/assignment_draft.json"):
                        st.success("✅ Assignment draft saved!")
    
    with tab3:
        st.markdown("### 📊 Evaluate & Release Scores")
        published_assignments = get_published_assignments()
        
        if not published_assignments:
            st.markdown("""
            <div class="custom-alert alert-info">
                <strong>📝 No Published Assignments</strong><br>
                Create and publish assignments first to see evaluation options here.
            </div>
            """, unsafe_allow_html=True)
        else:
            selected_assignment = st.selectbox(
                "Select Assignment to Evaluate",
                published_assignments,
                format_func=lambda x: f"📚 {x['config']['topic']} - {x['config']['level']} Level"
            )
            
            if selected_assignment:
                # Get all student responses for this assignment
                response_files = [f for f in os.listdir("data") if f.startswith(f"response_") and f.endswith(f"_{selected_assignment['id']}.json")]
                
                if not response_files:
                    st.markdown("""
                    <div class="custom-alert alert-warning">
                        <strong>⏳ No Student Responses</strong><br>
                        Students haven't submitted responses for this assignment yet.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"#### 📋 Student Responses ({len(response_files)} submissions)")
                    
                    # Evaluate all responses button
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("🤖 Evaluate All Responses", type="primary"):
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for idx, response_file in enumerate(response_files):
                                progress_bar.progress((idx + 1) / len(response_files))
                                response_data = load_json(f"data/{response_file}")
                                student_email = response_data['student_email']
                                status_text.text(f"Evaluating {get_username_from_email(student_email)}...")
                                
                                # Evaluate each question
                                detailed_results = []
                                total_score = 0
                                max_total = 0
                                
                                for item in selected_assignment['items']:
                                    student_answer = response_data['responses'].get(item['id'], '')
                                    
                                    # Use enhanced evaluation with error handling
                                    eval_result = evaluate_answer_with_ai(
                                        item['stem'],
                                        student_answer,
                                        item['answer_key'],
                                        item['type'],
                                        options=item.get('options') if item['type'] == 'mcq' else None
                                    )
                                    
                                    detailed_results.append({
                                        'question_id': item['id'],
                                        'question_type': item['type'],
                                        'question_text': item['stem'],
                                        'score': eval_result['score'],
                                        'max_score': eval_result['max_score'],
                                        'feedback': eval_result['feedback'],
                                        'student_answer': student_answer
                                    })
                                    
                                    total_score += eval_result['score']
                                    max_total += eval_result['max_score']
                                
                                # Calculate final results
                                percentage = (total_score / max_total * 100) if max_total > 0 else 0
                                
                                if percentage >= 90:
                                    grade = "A+"
                                elif percentage >= 85:
                                    grade = "A"
                                elif percentage >= 80:
                                    grade = "A-"
                                elif percentage >= 75:
                                    grade = "B+"
                                elif percentage >= 70:
                                    grade = "B"
                                elif percentage >= 65:
                                    grade = "B-"
                                elif percentage >= 60:
                                    grade = "C+"
                                elif percentage >= 55:
                                    grade = "C"
                                elif percentage >= 50:
                                    grade = "C-"
                                else:
                                    grade = "F"
                                
                                evaluation_results = {
                                    'total_score': total_score,
                                    'max_score': max_total,
                                    'percentage': round(percentage, 1),
                                    'grade': grade,
                                    'detailed_results': detailed_results,
                                    'overall_feedback': f"Overall performance: {'Outstanding' if percentage >= 90 else 'Excellent' if percentage >= 80 else 'Good' if percentage >= 70 else 'Satisfactory' if percentage >= 60 else 'Needs Improvement'}"
                                }
                                
                                # Save evaluation results
                                save_evaluation_results(student_email, selected_assignment['id'], evaluation_results)
                            
                            progress_bar.progress(1.0)
                            status_text.text("✅ All evaluations complete!")
                            st.success("✅ All responses evaluated successfully!")
                    
                    with col2:
                        scores_released = selected_assignment.get('scores_released', False)
                        if not scores_released:
                            if st.button("🚀 Release Scores", use_container_width=True):
                                if release_scores(selected_assignment['id']):
                                    st.success("✅ Scores released! Students can now view their results.")
                                    st.rerun()
                        else:
                            st.markdown("""
                            <div class="custom-alert alert-success">
                                <strong>✅ Scores Released</strong>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col3:
                        if st.button("🗑️ Delete Assignment", use_container_width=True):
                            if st.session_state.get('confirm_delete') != selected_assignment['id']:
                                st.session_state.confirm_delete = selected_assignment['id']
                                st.warning("⚠️ Click again to confirm deletion. This action cannot be undone!")
                            else:
                                if delete_assignment(selected_assignment['id']):
                                    st.success("✅ Assignment deleted successfully!")
                                    st.session_state.confirm_delete = None
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to delete assignment.")
                    
                    # Show evaluation results
                    st.markdown("#### 📊 Evaluation Results")
                    
                    results_data = []
                    for response_file in response_files:
                        response_data = load_json(f"data/{response_file}")
                        student_email = response_data['student_email']
                        
                        eval_file = f"data/evaluation_{student_email}_{selected_assignment['id']}.json"
                        if os.path.exists(eval_file):
                            eval_data = load_json(eval_file)
                            username = get_username_from_email(student_email)
                            results_data.append({
                                'Student': username,
                                'Email': student_email,
                                'Score': f"{eval_data['results']['total_score']}/{eval_data['results']['max_score']}",
                                'Percentage': f"{eval_data['results']['percentage']}%",
                                'Grade': eval_data['results']['grade']
                            })
                    
                    if results_data:
                        df = pd.DataFrame(results_data)
                        st.dataframe(df, use_container_width=True)
                        
                        # Export results
                        if st.button("📊 Export Results to CSV"):
                            csv_filename = f"results_{selected_assignment['config']['topic']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                            df.to_csv(f"data/{csv_filename}", index=False)
                            st.success(f"✅ Results exported to {csv_filename}")
    
    with tab4:
        st.markdown("### 📈 Analytics Dashboard")
        
        published_assignments = get_published_assignments()
        if not published_assignments:
            st.markdown("""
            <div class="custom-alert alert-info">
                <strong>📊 No Data Available</strong><br>
                Publish assignments and collect student responses to view analytics.
            </div>
            """, unsafe_allow_html=True)
        else:
            # Overall Statistics
            st.markdown("#### 📈 Overall Statistics")
            
            total_assignments = len(published_assignments)
            total_students = len(load_users().get("allowed_users", {}))
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div class="metric-container">
                    <div style="font-size: 1.5rem; font-weight: 600; color: #3b82f6;">{total_assignments}</div>
                    <div style="color: #6b7280;">Total Assignments</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-container">
                    <div style="font-size: 1.5rem; font-weight: 600; color: #3b82f6;">{total_students}</div>
                    <div style="color: #6b7280;">Registered Students</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Calculate response rate
            total_possible_responses = 0
            total_actual_responses = 0
            
            for assignment in published_assignments:
                total_possible_responses += total_students
                response_files = [f for f in os.listdir("data") if f.startswith(f"response_") and f.endswith(f"_{assignment['id']}.json")]
                total_actual_responses += len(response_files)
            
            response_rate = (total_actual_responses / total_possible_responses * 100) if total_possible_responses > 0 else 0
            
            with col3:
                st.markdown(f"""
                <div class="metric-container">
                    <div style="font-size: 1.5rem; font-weight: 600; color: #3b82f6;">{response_rate:.1f}%</div>
                    <div style="color: #6b7280;">Response Rate</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                scores_released_count = sum(1 for a in published_assignments if a.get('scores_released', False))
                st.markdown(f"""
                <div class="metric-container">
                    <div style="font-size: 1.5rem; font-weight: 600; color: #3b82f6;">{scores_released_count}</div>
                    <div style="color: #6b7280;">Scores Released</div>
                </div>
                """, unsafe_allow_html=True)


# Student Dashboard
else:
    st.markdown("### 📚 Your Learning Dashboard")
    published_assignments = get_published_assignments()
    
    if not published_assignments:
        st.markdown("""
        <div class="feature-card" style="text-align: center; padding: 3rem;">
            <div class="feature-icon">🔭</div>
            <div class="feature-title">No Assignments Available</div>
            <div class="feature-description">Check back later for new assignments from your professor.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        tab1, tab2 = st.tabs(["📝 Take Assignment", "📊 View Results"])
        
        with tab1:
            st.markdown("#### 🎯 Available Assignments")
            
            selected_assignment = st.selectbox(
                "Select Assignment", 
                published_assignments,
                format_func=lambda x: f"📚 {x['config']['topic']} - {x['config']['level']} Level ({len(x['items'])} questions)"
            )
            
            if selected_assignment:
                response_file = f"data/response_{st.session_state.user_email}_{selected_assignment['id']}.json"
                already_submitted = os.path.exists(response_file)
                
                if already_submitted:
                    st.markdown("""
                    <div class="custom-alert alert-success">
                        <strong>✅ Assignment Completed</strong><br>
                        You have already submitted this assignment. Great job!
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button("👀 Review Your Submission"):
                        response_data = load_json(response_file)
                        st.markdown("**📋 Your Responses:**")
                        for item in selected_assignment['items']:
                            with st.expander(f"{item['id']}: {item['stem'][:50]}..."):
                                st.write(f"**Question:** {item['stem']}")
                                st.write(f"**Your Answer:** {response_data['responses'].get(item['id'], 'No answer provided')}")
                else:
                    st.markdown("#### 📝 Answer the Questions Below")
                    
                    responses = {}
                    
                    for i, item in enumerate(selected_assignment['items']):
                        with st.container():
                            st.markdown(f"**Question {i+1}: {item['id']}** ({item['type'].upper()})")
                            st.markdown(item['stem'])
                            
                            if item['type'] == 'mcq':
                                # Display options for MCQ - handle both dict and list formats
                                options = item.get('options', {})
                                
                                # Convert list format to dict format if needed
                                if isinstance(options, list):
                                    if len(options) >= 4:
                                        options = {
                                            'A': options[0],
                                            'B': options[1], 
                                            'C': options[2],
                                            'D': options[3]
                                        }
                                    else:
                                        options = {'A': '', 'B': '', 'C': '', 'D': ''}
                                elif not isinstance(options, dict):
                                    options = {'A': '', 'B': '', 'C': '', 'D': ''}
                                
                                option_list = [
                                    f"A. {options.get('A', '')}",
                                    f"B. {options.get('B', '')}",
                                    f"C. {options.get('C', '')}",
                                    f"D. {options.get('D', '')}"
                                ]
                                
                                # Filter out empty options
                                valid_options = [opt for opt in option_list if opt.split('. ', 1)[1].strip()]
                                
                                if valid_options:
                                    response = st.radio(
                                        f"Select your answer:", 
                                        valid_options, 
                                        key=f"response_{item['id']}"
                                    )
                                    # Extract the selected option text (without the letter prefix)
                                    if response:
                                        selected_letter = response[0]  # A, B, C, or D
                                        responses[item['id']] = options.get(selected_letter, '')
                                    else:
                                        responses[item['id']] = ''
                                else:
                                    st.error("⚠️ No valid options available for this question")
                                    responses[item['id']] = ''
                            else:
                                response = st.text_area(
                                    f"Your answer:", 
                                    key=f"response_{item['id']}", 
                                    height=100,
                                    placeholder="Type your answer here..."
                                )
                                responses[item['id']] = response
                            
                            st.divider()
                    
                    if st.button("🚀 Submit Assignment", type="primary", use_container_width=True):
                        response_data = {
                            'student_email': st.session_state.user_email,
                            'assignment_id': selected_assignment['id'],
                            'responses': responses,
                            'submitted_at': datetime.now().isoformat()
                        }
                        
                        if save_json(response_data, response_file):
                            st.success("🎉 Assignment submitted successfully!")
                            st.balloons()
                            st.rerun()
        
        with tab2:
            st.markdown("#### 📊 Your Academic Performance")
            
            results = get_student_results(st.session_state.user_email)
            
            if not results:
                st.markdown("""
                <div class="custom-alert alert-info">
                    <strong>⏳ Results Pending</strong><br>
                    Your results will appear here once your professor releases the scores. Keep checking back!
                </div>
                """, unsafe_allow_html=True)
            else:
                # Overall performance summary
                total_assignments = len(results)
                total_percentage = sum(result["evaluation"]["results"]["percentage"] for result in results)
                average_percentage = total_percentage / total_assignments if total_assignments > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div style="font-size: 1.5rem; font-weight: 600; color: #1e40af;">{total_assignments}</div>
                        <div style="color: #6b7280;">Completed</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div style="font-size: 1.5rem; font-weight: 600; color: #1e40af;">{average_percentage:.1f}%</div>
                        <div style="color: #6b7280;">Average Score</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    best_score = max(result["evaluation"]["results"]["percentage"] for result in results)
                    st.markdown(f"""
                    <div class="metric-container">
                        <div style="font-size: 1.5rem; font-weight: 600; color: #1e40af;">{best_score:.1f}%</div>
                        <div style="color: #6b7280;">Best Score</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("#### 📋 Assignment Results")
                
                for result in results:
                    assignment = result["assignment"]
                    evaluation = result["evaluation"]
                    
                    with st.expander(f"📚 {assignment['config']['topic']} - Score: {evaluation['results']['percentage']}% ({evaluation['results']['grade']})"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(f"""
                            <div class="metric-container">
                                <div style="font-size: 1.5rem; font-weight: 600; color: #1e40af;">{evaluation['results']['total_score']}/{evaluation['results']['max_score']}</div>
                                <div style="color: #6b7280;">Score</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with col2:
                            st.markdown(f"""
                            <div class="metric-container">
                                <div style="font-size: 1.5rem; font-weight: 600; color: #1e40af;">{evaluation['results']['percentage']}%</div>
                                <div style="color: #6b7280;">Percentage</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with col3:
                            st.markdown(f"""
                            <div class="metric-container">
                                <div style="font-size: 1.5rem; font-weight: 600; color: #1e40af;">{evaluation['results']['grade']}</div>
                                <div style="color: #6b7280;">Grade</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("**📋 Detailed Question-wise Results:**")
                        for detail in evaluation['results']['detailed_results']:
                            with st.expander(f"{detail['question_id']} - {detail['score']}/{detail['max_score']} points"):
                                st.markdown(f"**Question:** {detail.get('question_text', 'N/A')}")
                                st.markdown(f"**Your Answer:** {detail['student_answer']}")
                                st.info(f"💡 **Feedback:** {detail['feedback']}")
                        
                        st.markdown("**🎯 Overall Feedback:**")
                        st.success(evaluation['results']['overall_feedback'])


# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #6b7280;">
    <div style="font-size: 1.1rem; font-weight: 500; margin-bottom: 0.5rem;">🌉 ShikshaSetu</div>
    <div>A bridge from effort to mastery • Built with ❤️ by Sajjan Singh</div>
    
</div>
""", unsafe_allow_html=True)
