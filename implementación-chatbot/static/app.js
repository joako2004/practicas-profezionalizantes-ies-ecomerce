// Definición de la clase Chatbox
class Chatbox {
    // Constructor de la clase, inicializa los elementos y estados
    constructor() {
        this.args = {
            openButton: document.querySelector('.chatbox__button'), // Botón para abrir el chat
            chatBox: document.querySelector('.chatbox__support'),   // Contenedor del chat
            sendButton: document.querySelector('.send__button')     // Botón para enviar mensajes
        }
        this.state = false; // Estado inicial del chatbox (cerrado)
        this.messages = []; // Lista para almacenar los mensajes del chat
    }

    // Método para agregar los eventos de interacción
    display() {
        const { openButton, chatBox, sendButton } = this.args;
      
        // Evento para abrir o cerrar el chat al hacer click en el botón
        openButton.addEventListener('click', () => this.toggleState(chatBox));
        
        // Evento para enviar un mensaje al hacer click en el botón de enviar
        sendButton.addEventListener('click', () => this.onSendButton(chatBox));
      
        // Evento para enviar el mensaje presionando la tecla Enter
        const node = chatBox.querySelector('input');
        node.addEventListener('keyup', ({ key }) => {
            if (key === 'Enter') {
                this.onSendButton(chatBox);
            }
        });
    }
      
    // Método para alternar el estado de apertura/cierre del chatbox
    toggleState(chatbox) {
        this.state = !this.state; // Cambia el estado de true a false o viceversa

        if (this.state) {
            chatbox.classList.add('chatbox--active'); // Muestra el chatbox
        } else {
            chatbox.classList.remove('chatbox--active'); // Oculta el chatbox
        }
    }

    // Método para manejar el envío de mensajes
    onSendButton(chatbox) {
        var textField = chatbox.querySelector('input'); // Obtiene el campo de entrada
        let text1 = textField.value; // Obtiene el texto ingresado
        
        if (text1 == "") {
            return; // No hace nada si el campo está vacío
        }

        // Guarda el mensaje del usuario
        let msg1 = { name: "User", message: text1 };
        this.messages.push(msg1);

        // Envia el mensaje al servidor para obtener una respuesta
        fetch($SCRIPT_ROOT + '/predict', {
            method: 'POST',
            body: JSON.stringify({ message: text1 }),
            mode: 'cors',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(r => r.json()) // Convierte la respuesta a JSON
        .then(r => {
            // Guarda la respuesta del servidor
            let msg2 = { name: 'SAM', message: r.answer };
            this.messages.push(msg2);
            this.updateChatText(chatbox); // Actualiza la ventana del chat
            textField.value = ''; // Limpia el campo de entrada
        })
        .catch((error) => {
            console.error('Error', error); // Muestra errores en consola
            this.updateChatText(chatbox); // Actualiza el chat aunque haya error
            textField.value = ''; // Limpia el campo de entrada
        });
    }

    // Método para actualizar la visualización de los mensajes en el chatbox
    updateChatText(chatbox) {
        var html = '';

        // Recorre los mensajes de forma invertida (último mensaje arriba)
        this.messages.slice().reverse().forEach(function(item, index) {
            if (item.name === "SAM") {
                html += '<div class="messages__item messages__item--visitor">' + item.message + '</div>';
            } else {
                html += '<div class="messages__item messages__item--operator">' + item.message + '</div>';
            }
        });

        // Inserta los mensajes en el contenedor correspondiente
        const chatmessage = chatbox.querySelector('.chatbox__messages');
        chatmessage.innerHTML = html;
    }
}

// Instancia la clase Chatbox y llama al método display para activar los eventos
const chatbox = new Chatbox();
chatbox.display();
