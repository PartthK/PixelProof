const express = require('express');
const ytdl = require('ytdl-core');
const cors = require('cors');
const app = express();
const port = 5500;

app.use(cors());
app.use(express.json());

app.post('/convert', async (req, res) => {
    const videoURL = req.body.link;
    if (!ytdl.validateURL(videoURL)) {
        return res.status(400).send('Invalid YouTube URL');
    }

    res.header('Content-Disposition', 'attachment; filename="video.mp4"');
    ytdl(videoURL, {
        format: 'mp4',
    }).pipe(res);
});

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});