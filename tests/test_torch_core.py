import torch


# Proves basic tensor creation, broadcasting, and matmul behave as expected.
def test_tensor_ops():
    # Create a 2x2 tensor to test basic operations.
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    # Create a 1D tensor that will broadcast across rows.
    b = torch.tensor([1.0, 2.0])
    # Add with broadcasting to verify shape and values.
    c = a + b

    # Ensure broadcasted shape matches the 2x2 input.
    assert c.shape == (2, 2)
    # Check that each row had b added correctly.
    assert torch.allclose(c, torch.tensor([[2.0, 4.0], [4.0, 6.0]]))

    # Multiply by a column vector to verify matmul behavior.
    d = torch.matmul(a, torch.tensor([[1.0], [1.0]]))
    # Confirm matmul produces a column vector.
    assert d.shape == (2, 1)
    # Validate the dot-products are correct.
    assert torch.allclose(d, torch.tensor([[3.0], [7.0]]))


# Proves autograd computes correct gradients for a simple scalar function.
def test_autograd_gradients():
    # Create inputs that track gradients.
    x = torch.tensor([2.0, -3.0], requires_grad=True)
    # Build a scalar loss to backpropagate from.
    y = (x ** 2).sum()
    # Run backward to populate gradients.
    y.backward()

    # Gradients should exist.
    assert x.grad is not None
    # d/dx of x^2 is 2x, so expect [4, -6].
    assert torch.allclose(x.grad, torch.tensor([4.0, -6.0]))


# Proves a single optimizer step can reduce loss on a tiny regression.
def test_single_step_optimization():
    # Fix randomness for deterministic weights.
    torch.manual_seed(0)

    # Create a tiny dataset: inputs of ones, targets of three.
    x = torch.ones(4, 1)
    y = torch.full((4, 1), 3.0)

    # Initialize parameters for a linear model.
    w = torch.randn(1, 1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    # Use mean squared error as the loss.
    loss_fn = torch.nn.MSELoss()
    # Use SGD for one optimization step.
    optimizer = torch.optim.SGD([w, b], lr=0.1)

    # Define the prediction function for the linear model.
    def predict(x_in: torch.Tensor) -> torch.Tensor:
        return x_in @ w + b

    # Compute loss before the update.
    loss_before = loss_fn(predict(x), y)
    # Clear gradients from any prior steps.
    optimizer.zero_grad()
    # Backpropagate to compute gradients.
    loss_before.backward()
    # Apply one parameter update.
    optimizer.step()

    # Compute loss after the update.
    loss_after = loss_fn(predict(x), y)
    # Expect the loss to decrease after the update.
    assert loss_after < loss_before
