document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('waterQualityForm');
    
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const formData = {
            rawSalinity: parseFloat(document.getElementById('rawSalinity').value),
            plantType: document.getElementById('plantType').value,
            plantLocation: document.getElementById('plantLocation').value,
            plantCapacity: parseFloat(document.getElementById('plantCapacity').value)
        };

        try {
            const response = await fetch('/calculate-water-quality', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });

            const data = await response.json();
            displayResults(data);
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while calculating water quality parameters.');
        }
    });
});

function displayResults(data) {
    // Show results card
    document.getElementById('resultsCard').style.display = 'block';
    
    // Update TDS
    document.getElementById('predictedTDS').textContent = data.predictedTDS;
    
    // Update WQI progress bar and value
    const wqiProgress = document.getElementById('wqiProgress');
    wqiProgress.style.width = `${data.wqi}%`;
    wqiProgress.setAttribute('aria-valuenow', data.wqi);
    document.getElementById('wqiValue').textContent = `WQI: ${data.wqi}`;
    
    // Set WQI color based on value
    if (data.wqi >= 80) {
        wqiProgress.className = 'progress-bar bg-success';
    } else if (data.wqi >= 60) {
        wqiProgress.className = 'progress-bar bg-info';
    } else if (data.wqi >= 40) {
        wqiProgress.className = 'progress-bar bg-warning';
    } else {
        wqiProgress.className = 'progress-bar bg-danger';
    }
    
    // Update suitability indicators
    const drinkingAlert = document.getElementById('drinkingAlert');
    const agricultureAlert = document.getElementById('agricultureAlert');
    
    // Drinking water suitability
    document.getElementById('drinkingWater').textContent = 
        data.suitability.drinking ? 'Suitable for Drinking' : 'Not Suitable for Drinking';
    drinkingAlert.className = `alert ${data.suitability.drinking ? 'alert-success' : 'alert-danger'}`;
    
    // Agricultural suitability
    document.getElementById('agriculture').textContent = 
        data.suitability.agriculture ? 'Suitable for Agriculture' : 'Not Suitable for Agriculture';
    agricultureAlert.className = `alert ${data.suitability.agriculture ? 'alert-success' : 'alert-danger'}`;
}
