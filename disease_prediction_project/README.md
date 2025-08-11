# Disease Prediction Project
This project utilizes machine learning models to predict diseases based on symptoms provided by users.


<div class="team">
            <h2>Meet Our Team</h2>
            <div class="team-member">
                <img src="https://via.placeholder.com/90" alt="Team Member 1">
                <h3>NAME </h3>
            </div>
            <div class="team-member">
                <img src="https://via.placeholder.com/90" alt="Team Member 2">
                <h3>NAME</h3>
               
            </div>
            <div class="team-member">
                <img src="https://via.placeholder.com/90" alt="Team Member 3">
                <h3>NAME</h3>
              
            </div>
            <div class="team-member">
                <img src="https://via.placeholder.com/90" alt="Team Member 4">
                <h3>NAME</h3>

            </div>


<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>VitalitasAI - Home</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            color: #fff;
            background-color: #f4f4f9;
            overflow: hidden; /* Prevent scroll */
        }

        /* Header styles */
        header {
            display: flex;
            justify-content: space-between; /* Align items to the left and right */
            align-items: center;
            width: 100%;
            padding: 20px;
            background-color: rgba(0, 0, 0, 0.8);
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.5);
            position: fixed;
            top: 0;
            z-index: 1000;
            color: white;
            font-weight: bold; /* Make header text bold */
        }

        .weblogo {
            height: 70px;
            width: 70px;
            background: url('https://thumbs.dreamstime.com/b/health-medical-care-icon-logo-dark-background-white-health-medical-care-icon-logo-dark-background-132228323.jpg') no-repeat center center / cover;
        }

        .header-title {
            font-size: 1.5em; /* Increased font size for the title */
            color: #4c86f9;
            text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.9);
        }

        nav.menu-bar {
            margin-left: auto; /* Push menu to the right */
        }

        nav.menu-bar ul {
            list-style: none;
            display: flex;
            padding: 0;
            margin: 0; /* Remove margin for better spacing */
        }

        nav.menu-bar ul li {
            margin: 0 15px; /* Adjusted margin for spacing */
        }

        nav.menu-bar ul li a {
            text-decoration: none;
            font-size: 18px;
            color: white;
            font-weight: bold;
            padding: 10px; /* Adjusted padding for better spacing */
            transition: color 0.3s;
        }

        nav.menu-bar ul li a:hover {
            color: #4c86f9;
            text-shadow: 0 0 10px white; /* Adding glow effect on hover */
        }

        /* Hero Section */
        .hero {
            position: relative;
            height: 100vh;
            background: url('https://d2jx2rerrg6sh3.cloudfront.net/images/news/ImageForNews_735641_16735286404943588.jpg') no-repeat center center/cover;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: flex-start; /* Align text to the left */
            padding: 20px; /* Added padding for left alignment */
            text-align: left; /* Changed text alignment */
            color: white;
        }

        .hero .overlay-text {
            font-size: 3.5em;
            font-weight: bold;
            text-shadow: 2px 2px 10px rgba(0, 0, 0, 0.8);
            line-height: 1.2; /* Improved line height for readability */
        }

        .hero .overlay-text span {
            display: block; /* Makes each line occupy a full row */
        }

        .hero .we-help {
            font-size: 3em; /* Bold heading size */
            font-weight: bold; /* Bold text */
            margin-bottom: 15px; /* Space below the heading */
        }

        /* Button styling */
        .btn-container {
            margin-top: 25px; /* Adjusted margin for button positioning */
            text-align: left; /* Align button to the left */
        }

        .btn {
            background-color: #4c86f9;
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 8px;
            font-size: 18px;
            cursor: pointer;
            text-decoration: none;
            transition: background-color 0.3s ease;
        }

        .btn:hover {
            background-color: #3b6fdb;
        }

        /* Info Section */
        .info-gallery {
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
            padding: 40px;
            margin-top: 20px;
            gap: 20px;
        }

        .info-container {
            background-color: rgba(0, 0, 0, 0.8);
            flex-basis: 45%;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
            text-align: left;
            transition: transform 0.3s; /* Smooth hover effect */
        }

        .info-container:hover {
            transform: scale(1.05); /* Slightly enlarge on hover */
        }

        .info-container h2 {
            font-size: 2em;
            margin-bottom: 15px;
            color: #4c86f9;
        }

        .info-container p {
            font-size: 1.1em;
            line-height: 1.6;
            color: #ddd;
        }

        footer {
            text-align: center;
            padding: 20px;
            background-color: rgba(0, 0, 0, 0.7);
            color: white;
            margin-top: 20px;
        }

        .site-title {
            font-size: 1.8em;
            font-weight: bold;
            margin: 0;
            color: #4c86f9; /* Title color */
        }

        #chatbot-button {
    position: fixed;
    bottom: 20px;
    right: 20px;
    background-color: #3660d4;
    color: white;
    padding: 35px 40px;
    border-radius: 50%;
    border: none;
    font-size: 20px;
    cursor: pointer;
    z-index: 1000;
    background: url('https://cdn-icons-png.flaticon.com/512/13330/13330989.png') no-repeat center center / cover;
    box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.3);
}

/* Reflection effect on hover */
#chatbot-button:hover::after {
    content: "";
    position: absolute;
    top: 100%;
    left: 0;
    right: 0;
    height: 35px;
    background: url('https://cdn-icons-png.flaticon.com/512/13330/13330989.png') no-repeat center center / cover;
    border-radius: 50%;
    opacity: 0.3;
    filter: blur(2px);
    transform: scaleY(-1);
    transition: opacity 0.3s ease; /* Smooth transition for the reflection */
}

/* Ensures reflection is hidden when not hovering */
#chatbot-button::after {
    content: "";
    opacity: 0; /* Hide by default */
    transition: opacity 0.3s ease;
}

        /* Chatbot container */
        #chatbot-container {
    display: none; /* Initially hidden */
    position: fixed;
    bottom: 70px; /* Distance from the bottom */
    right: 20px; /* Distance from the right */
    width: 400px; /* Increased width */
    height: 450px; /* Increased height */
    background-color: #f1f1f1;
    box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.3);
    border-radius: 8px;
    z-index: 2000;
    overflow: hidden;
}


        /* Chatbot header */
        #chatbot-header {
            background-color: #007bff;
            color: white;
            padding: 10px;
            font-size: 18px;
            text-align: center;
            cursor: pointer;
        }

        /* Chatbot body */
        #chatbot-body {
            padding: 10px;
            height: 320px;
            overflow-y: auto;
            background-color: white;
        }

        /* Chatbot footer */
        #chatbot-footer {
            display: flex;
            padding: 10px;
            background-color: #e9ecef;
        }

        #user-input {
            flex-grow: 1;
            padding: 10px;
        }

        button {
            background-color: #007bff;
            color: white;
            border: none;
            padding: 5px 10px;
            cursor: pointer;
        }

        /* Response styling */
        .bot-response, .user-message {
            margin: 5px 0;
            padding: 8px;
            border-radius: 4px;
            max-width: 80%;
        }
        .user-message {
            background-color: #007bff;
            color: white;
            align-self: flex-end;
        }
        .bot-response {
            background-color: #e9ecef;
            color: black;
            align-self: flex-start;
        }

         #popup-modal {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100vh;
  backdrop-filter: blur(4px); /* light blur */
  background-color: rgba(0, 0, 0, 0.3); /* subtle dark overlay */
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 3000;
  transition: all 0.3s ease;
}

#popup-content {
  background: linear-gradient(to bottom right, #ffffff, #f0f4ff);
  color: #1a1a1a;
  padding: 40px 30px;
  border-radius: 16px;
  max-width: 550px;
  width: 90%;
  text-align: center;
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.2);
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  animation: popupFadeIn 0.4s ease-out;
}

#popup-content h2 {
  margin-top: 0;
  font-size: 28px;
  color: #2563eb;
  font-weight: 600;
}

#popup-content p {
  margin-bottom: 18px;
  line-height: 1.6;
  font-size: 16px;
  color: #333;
}

#close-popup {
  background-color: #2563eb;
  color: white;
  padding: 12px 24px;
  border: none;
  border-radius: 8px;
  font-size: 16px;
  font-weight: 500;
  cursor: pointer;
  transition: background-color 0.3s ease;
}

#close-popup:hover {
  background-color: #1e4ed8;
}

/* Animation */
@keyframes popupFadeIn {
  from {
    transform: translateY(20px);
    opacity: 0;
  }
  to {
    transform: translateY(0px);
    opacity: 1;
  }
}

    </style>
</head>
<body>
<div id="popup-modal">
  <div id="popup-content">
    <h2>Welcome to VitalitasAI</h2>

    <p><strong>VitalitasAI</strong> helps you discover reliable healthcare solutions based on your symptoms and personal needs.</p>

    <p>We use advanced algorithms to guide you through symptom checks, dietary suggestions, wellness tips, and more — all in a conversational, easy-to-use format.</p>

    <h3>Key Features</h3>
    <ul>
      <li>AI-powered symptom checker</li>
      <li>Personalized diet and nutrition guidance</li>
      <li>Appointment and schedule suggestions</li>
      <li>Mental health support and resources</li>
      <li>Basic medication information (non-prescriptive)</li>
    </ul>

    <h3>How It Works</h3>
    <ol>
      <li>Type your symptoms or questions in the chatbot.</li>
      <li>VitalitasAI analyzes and responds with insights or suggestions.</li>
      <li>Use the tips to make better health decisions or connect with a doctor.</li>
    </ol>

    <h3>Your Privacy Matters</h3>
    <p>We do <strong>not</strong>share your personal data. All chats are anonymous and confidential.</p>

    <p><small>This app does not replace medical diagnosis. For emergencies or serious conditions, consult a healthcare professional immediately.</small></p>

    <button id="close-popup">Got it!</button>
  </div>
</div>

    <!-- Header -->
    <header>
        <div class="weblogo"></div>
        <div class="header-title">VitalitasAI</div>
        <nav class="menu-bar">
            <ul>
                <li><a href="/logout">Logout</a></li>
                <li><a href="/feedback">Feedback</a></li>
                <li><a href="/login">Login</a></li>
                <li><a href="/about">About Us</a></li>
            </ul>
        </nav>
    </header>

    <section class="hero">
        <div class="we-help">We Help!</div> <!-- Bold heading added here -->
        <div class="overlay-text">
            <span>Access Reliable</span>
            <span>Healthcare</span>
            <span>Solutions at Your Fingertips</span>
            <span>Get Started Today!</span>
        </div>
        <!-- Button to redirect to Prediction Page -->
        <div class="btn-container">
            <a href="/predict" class="btn">Go to MyClinic page</a>
        </div>
    </section>

    <!-- Info Section -->
    <section class="info-gallery">
        <div class="info-container">
            <h2>Comprehensive Health Information</h2>
            <p>Find trustworthy remedies and advice on various health symptoms, and stay informed.</p>
        </div>
        <div class="info-container">
            <h2>Connect with Healthcare Experts</h2>
            <p>Consult with qualified professionals for personalized health solutions and guidance.</p>
        </div>
    </section>

    <button id="chatbot-button"></button>

    <!-- Chatbot Container -->
    <div id="chatbot-container">
        <div id="chatbot-header">VitalitasAI</div>
        <div id="chatbot-body"></div>
        <div id="chatbot-footer">
            <input type="text" id="user-input" placeholder="Type your message..." required>
            <button onclick="sendMessage()">Send</button>
        </div>
    </div>

<script>
  document.addEventListener("DOMContentLoaded", () => {
    // ====== [1] POPUP MODAL HANDLING ======
    const popup = document.getElementById("popup-modal");
    const closeBtn = document.getElementById("close-popup");

    if (popup && closeBtn) {
      // Show popup once per session
      if (!sessionStorage.getItem("popupShown")) {
        popup.style.display = "flex";
        sessionStorage.setItem("popupShown", "true");
      } else {
        popup.style.display = "none";
      }

      // Close popup on click
      closeBtn.addEventListener("click", () => {
        popup.style.display = "none";
      });
    }

    // ====== [2] TOGGLE CHATBOT VISIBILITY ======
    const chatbotButton = document.getElementById('chatbot-button');
    const chatbotContainer = document.getElementById('chatbot-container');

    if (chatbotButton && chatbotContainer) {
      chatbotButton.addEventListener('click', () => {
        chatbotContainer.style.display =
          chatbotContainer.style.display === 'none' ? 'block' : 'none';
      });
    }

    // ====== [3] SEND MESSAGE TO CHATBOT ======
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');

    if (userInput && sendButton) {
      // Send on Enter key
      userInput.addEventListener("keypress", function (e) {
        if (e.key === "Enter") {
          e.preventDefault();
          sendMessage();
        }
      });

      // Send on button click
      sendButton.addEventListener("click", sendMessage);
    }

    // ====== [4] FUNCTION: Send message to backend and display ======
    function sendMessage() {
      const userMessage = userInput.value.trim();

      if (userMessage) {
        displayMessage(userMessage, 'user-message');
        userInput.value = '';  // Clear input

        // Send to server
        fetch('/get_response', {
          method: 'POST',
          headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
          body: new URLSearchParams({ user_input: userMessage })
        })
        .then(response => response.json())
        .then(data => {
          displayMessage(data.response || "⚠️ No response received.", 'bot-response');
        })
        .catch(error => {
          console.error('Error:', error);
          displayMessage("⚠️ Error communicating with server.", 'bot-response');
        });
      }
    }

    // ====== [5] FUNCTION: Display message ======
    function displayMessage(message, className) {
      const messageElement = document.createElement('div');
      messageElement.className = className;
      messageElement.textContent = message;
      const chatbotBody = document.getElementById('chatbot-body');
      chatbotBody.appendChild(messageElement);
      chatbotBody.scrollTop = chatbotBody.scrollHeight;
    }
  }); // <== ✅ Properly closed DOMContentLoaded function
</script> <!-- ✅ Properly closed script tag -->

</body>
</html>