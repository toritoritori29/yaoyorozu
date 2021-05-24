
import dataset

class Polygon:
    def __init__(self, points):
        self.points = points

def test_regmap():
    pol = Polygon([(3, 3), (7, 7)])
    regmap = dataset.create_regmap((10, 10), [pol], 0.5, 0.5)
    assert regmap.shape == (5, 5, 2), "Shape check failed."


def test_heatmap():
    pol = Polygon([(3, 3), (7, 7), (5, 5), (2, 2)])
    regmap = dataset.create_heatmap((10, 10), [pol], 4)
    assert regmap.shape == (10, 10, 4), "Shape check failed."

if __name__ == "__main__":
    test_heatmap()
    test_regmap()
    print("OK")


