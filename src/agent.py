import json
import time
from pydantic import ValidationError
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from messenger import Messenger
from data_loader import load_quora_dataset, sample_dataset
from schemas import QuestionPair, DeduplicationResponse, EvalRequest


class Agent:
    required_roles: list[str] = ["dedup_predictor"]
    required_config_keys: list[str] = ["sample_size", "random_seed"]

    def __init__(self):
        self.messenger = Messenger()
        self.dataset = None

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_roles = set(self.required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"

        missing_config_keys = set(self.required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"

        sample_size = request.config.get("sample_size")
        if not isinstance(sample_size, int) or sample_size <= 0:
            return False, "sample_size must be a positive integer"
        if sample_size > 1000:
            return False, "sample_size must be <= 1000"

        random_seed = request.config.get("random_seed")
        if not isinstance(random_seed, int):
            return False, "random_seed must be an integer"

        return True, "ok"

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Implement your agent logic here.

        Args:
            message: The incoming message
            updater: Report progress (update_status) and results (add_artifact)

        Use self.messenger.talk_to_agent(message, url) to call other agents.
        """
        input_text = get_message_text(message)

        try:
            request: EvalRequest = EvalRequest.model_validate_json(input_text)
            ok, msg = self.validate_request(request)
            if not ok:
                await updater.reject(new_agent_text_message(msg))
                return
        except ValidationError as e:
            await updater.reject(new_agent_text_message(f"Invalid request: {e}"))
            return

        purple_agent_url = str(request.participants["dedup_predictor"])
        sample_size = request.config["sample_size"]
        random_seed = request.config["random_seed"]

        await updater.update_status(
            TaskState.working, new_agent_text_message("Loading Quora dataset...")
        )

        try:
            if self.dataset is None:
                self.dataset = load_quora_dataset()
        except Exception as e:
            await updater.reject(new_agent_text_message(f"Failed to load dataset: {e}"))
            return

        samples = sample_dataset(self.dataset, sample_size, random_seed)

        await updater.update_status(
            TaskState.working, new_agent_text_message(f"Starting evaluation of {len(samples)} question pairs...")
        )

        predictions = []
        ground_truth = []
        justifications = []
        errors = []
        start_time = time.time()

        for idx, sample in enumerate(samples):
            if idx > 0 and idx % 20 == 0:
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"Progress: {idx}/{len(samples)} pairs evaluated")
                )

            question1 = sample["sentence1"]
            question2 = sample["sentence2"]
            is_duplicate = sample["label"]

            pair = QuestionPair(question1=question1, question2=question2)
            question_msg = pair.model_dump_json()

            try:
                response_text = await self.messenger.talk_to_agent(
                    message=question_msg,
                    url=purple_agent_url,
                    new_conversation=True,
                    timeout=30
                )

                prediction = None
                justification = None

                try:
                    response_data = json.loads(response_text)
                    response = DeduplicationResponse.model_validate(response_data)
                    prediction = response.prediction
                    justification = response.justification
                except (json.JSONDecodeError, ValidationError) as e:
                    errors.append({
                        "index": idx,
                        "error": f"Invalid A2A response format: {str(e)[:100]}"
                    })
                    continue

                if prediction is not None:
                    predictions.append(prediction)
                    ground_truth.append(is_duplicate)
                    if justification:
                        justifications.append({
                            "index": idx,
                            "justification": justification,
                            "prediction": prediction,
                            "actual": is_duplicate
                        })

            except Exception as e:
                errors.append({
                    "index": idx,
                    "error": str(e)
                })

        execution_time = time.time() - start_time

        if len(predictions) == 0:
            await updater.reject(new_agent_text_message("No valid predictions received from purple agent"))
            return

        correct_predictions = sum(p == g for p, g in zip(predictions, ground_truth))
        accuracy = correct_predictions / len(samples)

        precision = precision_score(ground_truth, predictions, zero_division=0)
        recall = recall_score(ground_truth, predictions, zero_division=0)
        f1 = f1_score(ground_truth, predictions, zero_division=0)

        successful_communications = len(predictions)

        results = {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "execution_time_seconds": round(execution_time, 2),
            "successful_communications": successful_communications
        }

        summary_text = f"""Quora Deduplication Evaluation Results

Accuracy: {accuracy:.2%} ({correct_predictions}/{len(samples)} correct)
Precision: {precision:.4f}
Recall: {recall:.4f}
F1 Score: {f1:.4f}
Execution Time: {execution_time:.2f} seconds
Successful Communications: {successful_communications}/{len(samples)}
"""

        await updater.add_artifact(
            parts=[
                Part(root=TextPart(text=summary_text)),
                Part(root=DataPart(data=results))
            ],
            name="Evaluation Results",
        )
