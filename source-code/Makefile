SUBDIRS = anomaly_detection Chess_game data_analysis_and_feature_engineering \
          deep_learning_basics deep_learning_category_theory deep_learning_image_generation \
          deep_learning_nlp knowledge_representation llm_local_models llm_public_apis \
          machine-learning neural_network_category_theory openknowledge_format \
          regression_and_clustering reinforcement_learning rete-algorithm \
          semantic_web_LLM symbolic-AI text-adventure-game

.PHONY: clean check test run $(SUBDIRS)

clean:
	for dir in $(SUBDIRS); do $(MAKE) -C $$dir clean; done
	rm -rf */.venv

check:
	for dir in $(SUBDIRS); do $(MAKE) -C $$dir check; done

test:
	for dir in $(SUBDIRS); do $(MAKE) -C $$dir test; done

run:
	for dir in $(SUBDIRS); do $(MAKE) -C $$dir run; done
