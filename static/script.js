document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('sudokuImage');
    const preview = document.getElementById('preview');
    const placeholder = document.getElementById('placeholder');
    const solveBtn = document.getElementById('solveBtn');
    const results = document.getElementById('results');
    const solutionGrid = document.getElementById('solutionGrid');

    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            preview.src = URL.createObjectURL(file);
            preview.style.display = 'block';
            placeholder.style.display = 'none';
            solveBtn.disabled = false;
            results.style.display = 'none';
        }
    });

    solveBtn.addEventListener('click', async () => {
        const file = fileInput.files[0];
        if (!file) {
            alert('Please select an image first');
            return;
        }

        solveBtn.disabled = true;
        solveBtn.textContent = 'Solving...';

        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch('/solve', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.error) {
                alert(`Error: ${data.error}`);
            } else {
                displaySolution(data.original, data.solution);
                results.style.display = 'block';
            }
        } catch (error) {
            alert('Failed to process image. Please try again.');
            console.error(error);
        } finally {
            solveBtn.disabled = false;
            solveBtn.textContent = 'Solve Sudoku';
        }
    });

    function displaySolution(original, solution) {
        let html = '<table class="sudoku-grid">';
        
        for (let i = 0; i < 9; i++) {
            html += '<tr>';
            for (let j = 0; j < 9; j++) {
                const classes = [];
                if (j === 2 || j === 5) classes.push('thick-right');
                if (i === 2 || i === 5) classes.push('thick-bottom');
                
                const cellValue = solution[i][j];
                const isOriginal = original[i][j] !== 0;
                
                if (isOriginal) {
                    classes.push('original');
                    html += `<td class="${classes.join(' ')}">${cellValue}</td>`;
                } else {
                    classes.push('solved');
                    html += `<td class="${classes.join(' ')}">${cellValue}</td>`;
                }
            }
            html += '</tr>';
        }
        
        html += '</table>';
        solutionGrid.innerHTML = html;
    }
});