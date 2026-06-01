from game.twister import TwisterMat


def test_twister_mat_size():
    mat = TwisterMat()
    assert len(mat.circles) == 24
    assert mat.circle_at(0, 0).color == "red"
