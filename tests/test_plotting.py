from unittest.mock import MagicMock, patch


from adipo_finder.plotting import Plotting, prediction_plot


@patch("adipo_finder.plotting.plt")
def test_plot_3_channel_image(mock_plt, binary_image):
    """Test 3 channel plot."""
    # Setup mock to return figure and axes
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    # Create dummy images
    img = binary_image
    inv = binary_image
    seg = binary_image

    Plotting.plot_3_channel_image(img, inv, seg)

    # Check if plot was created
    assert mock_plt.subplots.called
    assert mock_plt.show.called


@patch("adipo_finder.plotting.plt")
def test_plot_centroids(mock_plt, binary_image):
    """Test centroid plot."""
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    Plotting.plot_centroids(binary_image)

    assert mock_plt.subplots.called
    assert mock_plt.show.called


@patch("adipo_finder.plotting.plt")
def test_prediction_plot(mock_plt, binary_image, labeled_image):
    """Test prediction plot."""
    # This one returns 3 or 4 axes depending on GT
    mock_fig = MagicMock()

    # Mock for 3 axes
    [MagicMock() for _ in range(3)]
    # Mock for 4 axes
    [MagicMock() for _ in range(4)]

    # We need to change return value dynamically or just return a list that is large enough
    # but subplots usually returns array of axes.
    # Let's set side_effect
    def subplots_side_effect(*args, **kwargs):
        ncols = kwargs.get("ncols", 1)  # or args?
        # Prediction plot calls subplots(1, 3) or (1, 4)
        if len(args) >= 2:
            ncols = args[1]
        axes = [MagicMock() for _ in range(ncols)]
        return mock_fig, axes

    mock_plt.subplots.side_effect = subplots_side_effect

    # Test without GT
    prediction_plot(
        binary_image, labeled_image, labeled_image, gt_image=None, show_plot=True
    )
    assert mock_plt.subplots.called
    assert mock_plt.show.called

    # Test with GT
    prediction_plot(
        binary_image,
        labeled_image,
        labeled_image,
        gt_image=labeled_image,
        show_plot=True,
    )
    assert mock_plt.subplots.called
