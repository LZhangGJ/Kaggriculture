"""Inference-only copy of the frozen action-event v3 architecture."""

EVENT_CLASSES = (
    "STOP", "NONE_OR_KEEP", "RELEASE", "WHEAT", "CARROT", "TOMATO",
    "STRAWBERRY", "MELON", "GOOSE", "COW", "SHEEP",
)


def build_model(context_width, observation_width, resource_width, scale=1):
    import torch

    projection, scalar = 16 * scale, 4 * scale
    embedding, hidden = 4 * scale, 64 * scale
    resource_hidden, event_embedding = 64 * scale, 8 * scale

    class ActionEventStudent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.context = torch.nn.Linear(context_width, projection)
            self.observation = torch.nn.Linear(observation_width, projection)
            self.observation_length = torch.nn.Linear(1, scalar)
            self.token_embeddings = torch.nn.ModuleList(
                torch.nn.Embedding(size, embedding)
                for size in (7, 32, 32, 32, 64, 64, 4))
            self.token = torch.nn.Linear(24 + 7 * embedding, projection)
            self.begin = torch.nn.Linear(3 * projection + scalar, hidden)
            self.resource = torch.nn.Linear(resource_width, resource_hidden)
            self.cell = torch.nn.Embedding(100, event_embedding)
            self.stage = torch.nn.Embedding(2, event_embedding)
            self.previous = torch.nn.Embedding(len(EVENT_CLASSES) + 1, event_embedding)
            self.gru = torch.nn.GRUCell(
                resource_hidden + 3 * event_embedding, hidden)
            self.head = torch.nn.Linear(hidden, len(EVENT_CLASSES))

        def initial_hidden(self, context, observation, observation_length,
                           token_continuous, token_categories, token_count):
            positions = torch.arange(token_continuous.shape[1], device=context.device)[None]
            mask = positions < token_count[:, None]
            denominator = token_count.clamp_min(1).float()[:, None]
            continuous = (token_continuous * mask[:, :, None]).sum(1) / denominator
            embedded = [(table(values) * mask[:, :, None]).sum(1) / denominator
                        for table, values in zip(self.token_embeddings, token_categories)]
            token_state = torch.cat((continuous, *embedded), dim=1)
            state = torch.cat((
                torch.relu(self.context(context)),
                torch.relu(self.observation(observation)),
                torch.relu(self.observation_length(observation_length[:, None])),
                torch.relu(self.token(token_state)),
            ), dim=1)
            return torch.tanh(self.begin(state))

        def step(self, hidden, resources, cells, stages, previous, legal):
            inputs = torch.cat((torch.relu(self.resource(resources)), self.cell(cells),
                                self.stage(stages), self.previous(previous)), dim=1)
            hidden = self.gru(inputs, hidden)
            logits = self.head(hidden)
            return logits.masked_fill(~legal, -1e9), hidden

    return ActionEventStudent()
