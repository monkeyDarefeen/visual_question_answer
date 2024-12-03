from openai import OpenAI

#please use your own api_key
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)


def generate_questions_from_summary(summary):
    """
    Generate a fun, simple question based on the provided summary.
    """
    prompt = (
        f"Create 1 fun and simple question for a child (5-10 years old) based on this summary of an image. "
        f"The question should be fun, interesting, and easy to understand: {summary}"
    )

    completion = client.chat.completions.create(
        model="meta/llama-3.1-405b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        top_p=0.9,
        max_tokens=150
    )
    print
    # Extract and return the generated question
    return completion.choices[0].message.content.strip()


def generate_feedback(question, user_answer, correct_summary, is_correct):
    """
    Generate dynamic feedback based on the correctness of the answer.
    This feedback is fun and encouraging for children.
    """
    if is_correct:
        return "Hooray! 🎉 You got it right! Great job!"
    
    prompt = (
        f"Question: {question}\nUser's Answer: {user_answer}\nCorrect Summary: {correct_summary}\n"
        f"Generate a friendly, fun, and encouraging response that tells the child the answer was incorrect, "
        f"but provides the correct answer in a fun way."
    )

    completion = client.chat.completions.create(
        model="meta/llama-3.1-405b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        top_p=0.9,
        max_tokens=150
    )

    # Extract and return the feedback
    return completion.choices[0].message.content.strip()


def evaluate_answer(question, user_answer, correct_summary):
    """
    Evaluate the user's answer using the model to check if it's correct.
    """
    prompt = (
        f"Question: {question}\nUser's Answer: {user_answer}\nDoes the answer match the summary: {correct_summary}?\n"
        f"Answer 'yes' or 'no'."
    )

    completion = client.chat.completions.create(
        model="meta/llama-3.1-405b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        top_p=0.9,
        max_tokens=50
    )

    # Extract and interpret the response
    evaluation = completion.choices[0].message.content.strip()
    return evaluation.lower() == "yes"


def chatbot(summary_list):
    """
    Chatbot function for fun interaction with a child using image summaries.
    Asks questions one by one and evaluates answers.
    """
    print("Hey there! 😊 Let’s have some fun with these cool questions! Ready to play?")

    for summary in summary_list:
        # Generate a fun question based on the current summary
        question = generate_questions_from_summary(summary)
        print(f"\nQuestion: {question}")

        # Get user's answer
        user_answer = input("Your answer: ").strip()

        # Evaluate the answer using the model
        is_correct = evaluate_answer(question, user_answer, summary)

        # Generate dynamic feedback from the model based on the correctness
        feedback = generate_feedback(question, user_answer, summary, is_correct)

        # Print the feedback
        print(feedback)

        # Ask if the user wants to continue to the next question
        continue_response = input("Do you want to try another question? (yes/no): ").strip().lower()
        if continue_response != "yes":
            print("Thanks for playing! 🎉 See you next time!")
            break
