let currentChart = null;

function predictCost(event) {
    event.preventDefault();

    // Get input values
    const capacity = document.getElementById('capacity').value;
    const salinity = document.getElementById('salinity').value;
    const location = document.getElementById('location').value;
    const year = document.getElementById('year').value;
    const plantType = document.getElementById('plantType').value;
    const financing = document.getElementById('financing').value;

    // Validate inputs
    if (!capacity || !salinity || !location || !year || !plantType || !financing) {
        alert('Please fill in all fields');
        return;
    }

    // In a real implementation, this would call your ML model API
    // For demo purposes, we'll generate a random prediction
    const predictedCost = (capacity * 1000) + (Math.random() * 1000000);

    // Display the prediction
    document.getElementById('costPrediction').innerHTML = `
        <strong>Predicted Cost:</strong> $${predictedCost.toLocaleString('en-US', {maximumFractionDigits: 2})}
        <br><br>
        <strong>Input Parameters:</strong><br>
        Capacity: ${capacity} m³/day<br>
        Salinity: ${salinity} mg/L<br>
        Location: ${location}<br>
        Year: ${year}<br>
        Plant Type: ${plantType}<br>
        Financing: ${financing}
    `;

    // Update chart
    updateChart(predictedCost);
}

function updateChart(predictedCost) {
    const ctx = document.getElementById('resultChart').getContext('2d');

    if (currentChart) {
        currentChart.destroy();
    }

    // Create sample cost breakdown
    const equipmentCost = predictedCost * 0.4;
    const laborCost = predictedCost * 0.3;
    const materialsCost = predictedCost * 0.2;
    const otherCosts = predictedCost * 0.1;

    currentChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Equipment', 'Labor', 'Materials', 'Other'],
            datasets: [{
                data: [equipmentCost, laborCost, materialsCost, otherCosts],
                backgroundColor: [
                    '#3498db',
                    '#2ecc71',
                    '#f1c40f',
                    '#e74c3c'
                ]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
}
