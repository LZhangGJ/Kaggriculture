class Market:

    def __init__(self, parser):

        self.parser = parser
        self.prices = parser.prices
        self.inventory = parser.inventory

class FakeParser:

    def __init__(self):

        self.prices = {
            "WHEAT": 120,
            "CARROT": 180,
        }

        self.inventory = {}