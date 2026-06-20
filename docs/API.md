# API Documentation

The SignBridge AI backend provides several endpoints for real-time sign language recognition, face enrollment, and data management.

## Endpoints

### 1. `GET /`

- **Description**: Health check and basic system info.
- **Response**: `200 OK`

### 2. `POST /api/recognize`

- **Description**: Process a video frame and return recognition results.
- **Payload**: Base64 encoded image frame.
- **Response**: JSON containing detected person, sign, and emotion.

### 3. `POST /api/enroll`

- **Description**: Enroll a new face into the system.
- **Payload**: Name, optional notes, and face landmarks.
- **Response**: Success/Failure status.

### 4. `GET /api/logs`

- **Description**: Retrieve conversation history.
- **Response**: List of conversation logs.

## Socket.io Events

### `frame` (Client -> Server)

- **Data**: Base64 string of the current video frame.

### `recognition_result` (Server -> Client)

- **Data**: JSON object with `face`, `gesture`, and `emotion` results.
