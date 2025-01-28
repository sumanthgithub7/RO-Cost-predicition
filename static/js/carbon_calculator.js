document.addEventListener('DOMContentLoaded', function() {
    const carbonCalculator = document.getElementById('carbonCalculator');
    const carbonResult = document.getElementById('carbonResult');
    const emissionsChart = document.getElementById('emissionsChart').getContext('2d');
    let chart = null;

    // Energy consumption per m³ of water (kWh per m³)
    const energyConsumption = {
        groundwater: 0.5,
        surface: 0.3,
        brackish: 1.0,
        seawater: 3.5
    };

    // Emission factors (kg CO₂ per kWh)
    const emissionFactors = {
        grid: 0.5,
        solar: 0.05,
        wind: 0.02,
        hybrid: 0.275
    };

    function calculateEmissions(waterType, capacity, energySource) {
        const dailyEnergy = capacity * energyConsumption[waterType];
        const dailyEmissions = dailyEnergy * emissionFactors[energySource];
        const annualEnergy = dailyEnergy * 365;
        const annualEmissions = dailyEmissions * 365;

        return {
            dailyEnergy,
            dailyEmissions,
            annualEnergy,
            annualEmissions
        };
    }

    function updateChart(waterType, capacity) {
        const energySources = Object.keys(emissionFactors);
        const emissions = energySources.map(source => {
            return calculateEmissions(waterType, capacity, source).dailyEmissions;
        });

        if (chart) {
            chart.destroy();
        }

        chart = new Chart(emissionsChart, {
            type: 'bar',
            data: {
                labels: energySources.map(source => source.charAt(0).toUpperCase() + source.slice(1)),
                datasets: [{
                    label: 'Daily Carbon Emissions (kg CO₂)',
                    data: emissions,
                    backgroundColor: [
                        'rgba(255, 99, 132, 0.5)',   // Grid
                        'rgba(255, 205, 86, 0.5)',   // Solar
                        'rgba(75, 192, 192, 0.5)',   // Wind
                        'rgba(54, 162, 235, 0.5)'    // Hybrid
                    ],
                    borderColor: [
                        'rgb(255, 99, 132)',
                        'rgb(255, 205, 86)',
                        'rgb(75, 192, 192)',
                        'rgb(54, 162, 235)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Carbon Emissions (kg CO₂/day)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }

    carbonCalculator.addEventListener('submit', function(e) {
        e.preventDefault();
        showLoading();

        const waterType = document.getElementById('waterType').value;
        const capacity = parseFloat(document.getElementById('dailyCapacity').value);
        const energySource = document.getElementById('energySource').value;

        if (!waterType || !capacity || !energySource || capacity <= 0) {
            hideLoading();
            carbonResult.innerHTML = `
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-triangle"></i> Please fill all fields with valid values.
                </div>
            `;
            carbonResult.style.display = 'block';
            return;
        }

        const results = calculateEmissions(waterType, capacity, energySource);

        setTimeout(() => {
            carbonResult.className = 'alert alert-success mt-3';
            carbonResult.innerHTML = `
                <h5><i class="fas fa-check-circle"></i> Carbon Footprint Results</h5>
                <div class="table-responsive">
                    <table class="table table-bordered mb-0">
                        <tbody>
                            <tr>
                                <th>Daily Energy Consumption:</th>
                                <td>${results.dailyEnergy.toFixed(2)} kWh</td>
                            </tr>
                            <tr>
                                <th>Daily Carbon Emissions:</th>
                                <td>${results.dailyEmissions.toFixed(2)} kg CO₂</td>
                            </tr>
                            <tr>
                                <th>Annual Energy Consumption:</th>
                                <td>${results.annualEnergy.toFixed(2)} kWh</td>
                            </tr>
                            <tr>
                                <th>Annual Carbon Emissions:</th>
                                <td>${results.annualEmissions.toFixed(2)} kg CO₂</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            `;
            carbonResult.style.display = 'block';

            // Update the emissions comparison chart
            updateChart(waterType, capacity);
            hideLoading();

            // Scroll to results
            carbonResult.scrollIntoView({ behavior: 'smooth' });
        }, 1000);
    });

    function showLoading() {
        document.getElementById('loading').style.display = 'flex';
    }

    function hideLoading() {
        document.getElementById('loading').style.display = 'none';
    }
});
