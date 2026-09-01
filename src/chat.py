from model_runtime import configure_utf8_console
from search_chunks import answer_question_with_details


EXIT_COMMANDS = {"exit", "quit", "salir"}
def main() -> None:
    configure_utf8_console()
    print("Chemistry RAG started. Type 'exit' or 'salir' to stop.")

    while True:
        question = input("\nQuestion: ").strip()

        if question.casefold() in EXIT_COMMANDS:
            print("Session ended.")
            break

        if not question:
            print("Enter a question.")
            continue

        try:
            print("Processing...", flush=True)
            result = answer_question_with_details(question)
            print(f"\nAnswer:\n{result.answer}")
        except KeyboardInterrupt:
            print("\nSession ended.")
            break
        except Exception as error:
            print(f"\nThe question could not be processed: {error}")


if __name__ == "__main__":
    main()
