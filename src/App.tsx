import "./App.css"
import Container from "./components/Container"

function App() {

  return (
    <>
      <h6>Auto grad viz
        <span> - A tool for visualizing automatic differentiation</span>
      </h6>
      <div className="card">
        <Container />
      </div>
    </>
  )
}

export default App
