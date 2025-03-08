document.addEventListener('DOMContentLoaded', function() {
    // Hide results sections initially
    document.getElementById('summarySection').style.display = 'none';
    document.getElementById('chartsSection').style.display = 'none';
    document.getElementById('negativeSection').style.display = 'none';
    document.getElementById('transcriptSection').style.display = 'none';
    
    // Show processing status when analyze button is clicked
    document.getElementById('analyzeBtn').addEventListener('click', function() {
        const audioFile = document.getElementById('audioFile').files[0];
        if (!audioFile) {
            alert('Please select an audio file to analyze');
            return;
        }
        
        // Show processing status
        document.getElementById('processing').style.display = 'block';
        
        // Simulate processing delay (in a real app, this would be an actual API call)
        setTimeout(function() {
            // Hide processing status
            document.getElementById('processing').style.display = 'none';
            
            // Show results sections
            document.getElementById('summarySection').style.display = 'block';
            document.getElementById('chartsSection').style.display = 'block';
            document.getElementById('negativeSection').style.display = 'block';
            document.getElementById('transcriptSection').style.display = 'block';
            
            // Load mock data and render charts
            loadMockData();
        }, 2000);
    });
    
    // Transcript filter buttons
    document.getElementById('showAllBtn').addEventListener('click', function() {
        filterTranscript('all');
        setActiveButton(this);
    });
    
    document.getElementById('showAgentBtn').addEventListener('click', function() {
        filterTranscript('agent');
        setActiveButton(this);
    });
    
    document.getElementById('showCustomerBtn').addEventListener('click', function() {
        filterTranscript('customer');
        setActiveButton(this);
    });
    
    document.getElementById('showNegativeBtn').addEventListener('click', function() {
        filterTranscript('negative');
        setActiveButton(this);
    });
});

function setActiveButton(button) {
    // Remove active class from all buttons
    document.querySelectorAll('.transcript-controls button').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Add active class to clicked button
    button.classList.add('active');
}

function filterTranscript(filter) {
    const transcriptLines = document.querySelectorAll('.transcript-line');
    
    transcriptLines.forEach(line => {
        if (filter === 'all') {
            line.style.display = 'block';
        } else if (filter === 'agent' && line.classList.contains('agent')) {
            line.style.display = 'block';
        } else if (filter === 'customer' && line.classList.contains('customer')) {
            line.style.display = 'block';
        } else if (filter === 'negative' && line.classList.contains('negative')) {
            line.style.display = 'block';
        } else {
            line.style.display = 'none';
        }
    });
}

function loadMockData() {
    // Load summary metrics
    document.getElementById('overallScore').textContent = '78%';
    document.getElementById('overallSentiment').textContent = 'Neutral';
    document.getElementById('avgResponseTime').textContent = '3.2s';
    document.getElementById('accuracyScore').textContent = '85%';
    
    // Load negative statements
    const negativeStatements = [
        'I\'ve been waiting for a resolution for over a week now.',
        'This is completely unacceptable and I want to speak to a manager.',
        'You\'re not listening to what I\'m saying at all.'
    ];
    
    const negativeList = document.getElementById('negativeStatements');
    negativeStatements.forEach(statement => {
        const li = document.createElement('li');
        li.textContent = statement;
        negativeList.appendChild(li);
    });
    
    // Load long pauses
    const longPauses = [
        'Pause of 8.3 seconds at 2:15',
        'Pause of 6.7 seconds at 4:32',
        'Pause of 5.1 seconds at 7:18'
    ];
    
    const pausesList = document.getElementById('longPauses');
    longPauses.forEach(pause => {
        const li = document.createElement('li');
        li.textContent = pause;
        pausesList.appendChild(li);
    });
    
    // Load transcript
    const transcriptData = [
        { time: '0:05', speaker: 'Agent', text: 'Thank you for calling customer support. My name is Michael. How can I help you today?', sentiment: 'positive' },
        { time: '0:12', speaker: 'Customer', text: 'Hi, I\'m calling about my recent order that hasn\'t arrived yet.', sentiment: 'neutral' },
        { time: '0:18', speaker: 'Agent', text: 'I\'m sorry to hear that. Can you please provide your order number so I can look into this for you?', sentiment: 'positive' },
        { time: '0:25', speaker: 'Customer', text: 'Yes, it\'s ABC123456.', sentiment: 'neutral' },
        { time: '0:40', speaker: 'Agent', text: 'Thank you. I can see your order was shipped three days ago. According to our tracking, it should arrive by tomorrow.', sentiment: 'positive' },
        { time: '0:48', speaker: 'Customer', text: 'But I paid for express shipping. It was supposed to arrive yesterday.', sentiment: 'negative' },
        { time: '1:02', speaker: 'Agent', text: 'I see that now. You\'re right, and I apologize for the delay. Let me check what happened.', sentiment: 'neutral' },
        { time: '1:35', speaker: 'Agent', text: 'It looks like there was a delay at the shipping facility. I\'m very sorry about this.', sentiment: 'neutral' },
        { time: '1:42', speaker: 'Customer', text: 'This is completely unacceptable. I paid extra for express shipping and now my package is late.', sentiment: 'negative' },
        { time: '1:50', speaker: 'Agent', text: 'I understand your frustration. I\'d like to offer you a refund for the shipping costs and a 15% discount on your next order.', sentiment: 'positive' },
        { time: '2:05', speaker: 'Customer', text: 'Well, I guess that helps a little. But I really needed that package yesterday for an event.', sentiment: 'neutral' },
        { time: '2:15', speaker: 'Agent', text: '...', sentiment: 'negative' },
        { time: '2:23', speaker: 'Agent', text: 'I completely understand. I\'ll also expedite a complaint to our shipping department to prevent this from happening again.', sentiment: 'positive' },
        { time: '2:30', speaker: 'Customer', text: 'Alright, thank you for your help.', sentiment: 'positive' },
        { time: '2:35', speaker: 'Agent', text: 'Is there anything else I can assist you with today?', sentiment: 'positive' },
        { time: '2:40', speaker: 'Customer', text: 'No, that\'s all. Thank you.', sentiment: 'positive' },
        { time: '2:45', speaker: 'Agent', text: 'Thank you for calling. Have a great day!', sentiment: 'positive' }
    ];
    
    const transcript = document.getElementById('transcript');
    transcriptData.forEach(line => {
        const div = document.createElement('div');
        div.className = `transcript-line ${line.speaker.toLowerCase()}`;
        
        if (line.sentiment === 'negative') {
            div.classList.add('negative');
        }
        
        const timeSpan = document.createElement('span');
        timeSpan.className = 'time';
        timeSpan.textContent = line.time;
        
        const speakerDiv = document.createElement('div');
        speakerDiv.className = 'speaker';
        speakerDiv.textContent = line.speaker;
        
        const textDiv = document.createElement('div');
        textDiv.textContent = line.text;
        
        div.appendChild(timeSpan);
        div.appendChild(speakerDiv);
        div.appendChild(textDiv);
        
        transcript.appendChild(div);
    });
    
    // Render charts
    renderSentimentChart();
    renderSentimentPieChart();
    renderResponseTimeChart();
    renderMetricsRadarChart();
}

function renderSentimentChart() {
    const ctx = document.getElementById('sentimentChart').getContext('2d');
    
    // Mock data for sentiment progression
    const labels = ['0:00', '0:30', '1:00', '1:30', '2:00', '2:30', '3:00'];
    const data = [0.6, 0.4, -0.2, -0.8, -0.3, 0.5, 0.7]; // -1 to 1 scale
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Sentiment Score',
                data: data,
                borderColor: '#4a6bff',
                backgroundColor: 'rgba(74, 107, 255, 0.1)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            scales: {
                y: {
                    min: -1,
                    max: 1,
                    ticks: {
                        callback: function(value) {
                            if (value === 1) return 'Positive';
                            if (value === 0) return 'Neutral';
                            if (value === -1) return 'Negative';
                            return '';
                        }
                    }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            let value = context.raw;
                            let sentiment = 'Neutral';
                            if (value > 0.3) sentiment = 'Positive';
                            if (value < -0.3) sentiment = 'Negative';
                            return `Sentiment: ${sentiment} (${value.toFixed(2)})`;
                        }
                    }
                }
            }
        }
    });
}

function renderSentimentPieChart() {
    const ctx = document.getElementById('sentimentPieChart').getContext('2d');
    
    // Mock data for sentiment distribution
    const data = {
        labels: ['Positive', 'Neutral', 'Negative'],
        datasets: [{
            data: [60, 25, 15],
            backgroundColor: ['#52c41a', '#faad14', '#ff4d4d'],
            borderWidth: 0
        }]
    };
    
    new Chart(ctx, {
        type: 'pie',
        data: data,
        options: {
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.label}: ${context.raw}%`;
                        }
                    }
                }
            }
        }
    });
}

function renderResponseTimeChart() {
    const ctx = document.getElementById('responseTimeChart').getContext('2d');
    
    // Mock data for response times
    const labels = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6'];
    const data = [2.1, 3.5, 8.3, 2.8, 1.9, 2.2]; // in seconds
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Response Time (seconds)',
                data: data,
                backgroundColor: data.map(value => {
                    if (value < 3) return '#52c41a'; // Good
                    if (value < 5) return '#faad14'; // Warning
                    return '#ff4d4d'; // Bad
                })
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Seconds'
                    }
                }
            }
        }
    });
}

function renderMetricsRadarChart() {
    const ctx = document.getElementById('metricsRadarChart').getContext('2d');
    
    // Mock data for key metrics
    const data = {
        labels: [
            'Accuracy',
            'Empathy',
            'Solution Quality',
            'Responsiveness',
            'Knowledge',
            'Professionalism'
        ],
        datasets: [{
            label: 'Agent Performance',
            data: [85, 70, 75, 60, 90, 80],
            backgroundColor: 'rgba(74, 107, 255, 0.2)',
            borderColor: '#4a6bff',
            pointBackgroundColor: '#4a6bff',
            pointBorderColor: '#fff',
            pointHoverBackgroundColor: '#fff',
            pointHoverBorderColor: '#4a6bff'
        }]
    };
    
    new Chart(ctx, {
        type: 'radar',
        data: data,
        options: {
            scales: {
                r: {
                    angleLines: {
                        display: true
                    },
                    suggestedMin: 0,
                    suggestedMax: 100
                }
            }
        }
    });
}