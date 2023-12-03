# Machine Learning - Kubernetes

Kubernetes, K8s, is an open-source container orchestration platform designed to automate the deployment, scaling, and management of containerized applications. It was originally developed by Google and is now maintained by the Cloud Native Computing Foundation (CNCF). Kubernetes provides a robust and flexible environment for deploying, managing, and scaling applications.

### Key Concepts in Kubernetes:

1. **Containerization:**
   - Kubernetes works with containerized applications. Containers encapsulate an application and its dependencies, ensuring consistency across different environments.

2. **Nodes:**
   - A Kubernetes cluster consists of a set of machines called nodes. Nodes can be physical or virtual machines, and they run containerized applications managed by Kubernetes.

3. **Pods:**
   - The smallest deployable units in Kubernetes are pods. A pod represents a single instance of a running process in a cluster and may contain one or more containers.

4. **Services:**
   - Services provide a stable endpoint for accessing a set of pods. They allow for load balancing and abstract the underlying pod infrastructure, enabling seamless communication between different parts of an application.

5. **Deployments:**
   - Deployments define the desired state for applications, including the number of replicas. Kubernetes ensures that the actual state matches the desired state, automatically scaling up or down as needed.

6. **ConfigMaps and Secrets:**
   - ConfigMaps hold configuration data, while Secrets hold sensitive information such as passwords or API keys. These can be injected into pods to configure applications.

7. **Replication Controllers:**
   - Replication Controllers ensure that a specified number of replicas of a pod are running at all times. Deployments are a higher-level abstraction that provides more functionality.

8. **Namespace:**
   - Namespaces provide a way to divide cluster resources between multiple users or teams, helping to organize and isolate resources.

9. **Kubelet and Kube-proxy:**
   - Kubelet is an agent that runs on each node, responsible for ensuring that containers are running in a pod. Kube-proxy maintains network rules on nodes.

### Key Features of Kubernetes:

1. **Container Orchestration:**
   - Kubernetes automates the deployment, scaling, and management of containerized applications, allowing for efficient resource utilization.

2. **Declarative Configuration:**
   - Kubernetes allows users to define the desired state of their applications and infrastructure, and it continuously works to ensure the current state matches the desired state.

3. **Service Discovery and Load Balancing:**
   - Kubernetes provides built-in mechanisms for service discovery and load balancing, making it easy for applications to communicate with each other.

4. **Rolling Updates and Rollbacks:**
   - Applications can be updated seamlessly with rolling updates, and rollbacks can be performed if issues arise.

5. **Self-healing:**
   - Kubernetes monitors the health of applications and automatically restarts failed containers or replaces failed pods.

Kubernetes is widely used in cloud-native application development and is a fundamental technology for building scalable and resilient microservices architectures. It abstracts away many of the complexities associated with deploying and managing containerized applications at scale.

kubectl and kind are both essential tools in the Kubernetes ecosystem, serving different purposes but often used together for development and testing. 


## kubectl:

kubectl is the command-line tool for interacting with Kubernetes clusters. It allows you to deploy and manage applications, inspect and manage cluster resources, and view logs.

**Key Features:**
1. **Resource Management:** `kubectl` enables you to create, modify, and delete Kubernetes resources (pods, services, deployments, etc.).
2. **Cluster Information:** You can get information about your cluster, including nodes, pods, and services.
3. **Interactivity:** You can execute commands inside running containers, copy files to/from containers, and open a shell in a running pod.

**Example Commands:**
- `kubectl get pods`: List all pods in the current namespace.
- `kubectl create deployment <name> --image=<image>`: Deploy an application.
- `kubectl logs <pod-name>`: View logs for a specific pod.

## kind (Kubernetes in Docker):

**What is it?**

`kind` stands for Kubernetes IN Docker. It's a tool for running local Kubernetes clusters using Docker container "nodes." It's lightweight and easy to set up.

**Key Features:**
1. **Local Development:** `kind` is designed for creating Kubernetes clusters on your local machine, which is useful for development and testing.
2. **Containerized Nodes:** Each node in the Kubernetes cluster is a Docker container, making it quick to spin up and down.
3. **Cluster Lifecycle:** You can create, delete, and manage entire Kubernetes clusters with `kind`.

**Example Commands:**

- `kind create cluster`: Create a new Kubernetes cluster.
- `kind delete cluster`: Delete the cluster created with `kind`.
- `kind get clusters`: List all running `kind` clusters.

**Use Case:**
- Developers often use `kind` to create local Kubernetes clusters for testing and debugging without the need for a full-scale cluster.

**Combining kubectl and kind:**
1. **Setup Cluster:** Use `kind` to create a local cluster.
2. **Configure kubectl:** Set `kubectl` to use the `kind` cluster with `kubectl config use-context <cluster-name>`.

By combining these tools, you can efficiently develop, test, and experiment with Kubernetes configurations and applications locally before deploying to a production environment.

## Create a Cluster

- Create a cluster by running this command
  
```bash
kind create cluster
```

- Verify the cluster information

```bash
kubectl cluster-info

# output
Kubernetes control plane is running at https://127.0.0.1:36891

CoreDNS is running at https://127.0.0.1:36891/api/v1/namespaces/kube-system/services/kube-dns:dns/proxy

```

The endpoints provided by kubectl cluster-info are primarily for configuration and informational purposes, helping you understand where the essential components of your Kubernetes cluster are running. They don't directly represent endpoints for the applications.

- Kubernetes control plane is running at https://127.0.0.1:36891:

This line indicates the address where the Kubernetes control plane is running. In your case, it's running locally on (127.0.0.1) on port 36891. The control plane includes components like the API server, controller manager, scheduler, and etcd.

- CoreDNS is running at https://127.0.0.1:36891/api/v1/namespaces/kube-system/services/kube-dns:dns/proxy:

CoreDNS is the default DNS (Domain Name System) provider in Kubernetes clusters. It's responsible for DNS-based service discovery within the cluster. The URL provided is the endpoint where CoreDNS is reachable. 

The URL structure is a proxy endpoint for the CoreDNS service in the kube-system namespace. This endpoint is used by other components in the cluster to resolve service names to their corresponding IP addresses.

## Get a list of running services

To list the running services in a Kubernetes cluster using `kubectl`, you can use the following command:

```bash
kubectl get services
```

This command provides a list of all services running in the default namespace. If you have services in a specific namespace, you can use the `-n` or `--namespace` flag to specify the namespace:

```bash
kubectl get services -n <namespace>
```

The output will display information about each service, including its name, type, cluster IP, external IP (if applicable), ports, and age.

For example, the output might look like this:

```
NAME         TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)   AGE
my-service   NodePort    10.101.143.127  <none>        80:30080/TCP  2d
```

Here:
- `NAME` is the name of the service.
- `TYPE` is the type of service (NodePort, ClusterIP, LoadBalancer, etc.).
- `CLUSTER-IP` is the internal IP address assigned to the service.
- `EXTERNAL-IP` is the external IP address (if applicable, it might show `<none>` for services without external access).
- `PORT(S)` shows the mapping of ports (service port and target port).
- `AGE` indicates how long the service has been running.

This information is helpful for understanding the services in your cluster and how they are configured.

## Register a Docker image with kind

To use a Docker image with `kind` (Kubernetes IN Docker), you'll need to follow these general steps:

1. **Load Docker Image into Kind Nodes:**
   - Use the `kind load docker-image` command to load your Docker image into the nodes of the `kind` cluster. This makes the image available to the nodes for running containers.

   ```bash
   kind load docker-image <your-image-name>
   ```

   Replace `<your-image-name>` with the name of your Docker image.

2. **Update Kubernetes Manifests:**
   - Update your Kubernetes manifests (e.g., Deployment or Pod YAML files) to use the Docker image you loaded into the `kind` cluster. Update the `image` field with the image name you provided.

   For example:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: credit
spec:
  selector:
    matchLabels:
      app: credit
  replicas: 1
  template:
    metadata:
      labels:
        app: credit
    spec:
      containers:
      - name: credit
        image: ozkary-credit:latest
        resources:
          requests:
            memory: "64Mi"
            cpu: "100m"
          limits:
            memory: "128Mi"  # Adjust based on your application's requirements
            cpu: "200m"     # Adjust based on your application's requirements
        ports:
        - containerPort: 9696
```

   Configuring resource requests and limits for a container in Kubernetes involves specifying the amount of CPU and memory that the container is allowed to use. Here's how you can adjust your Deployment YAML for the `ozkary-credit` Docker image:

Explanation of changes:

- **`image`**: Replace `<Image>` with the actual name of your Docker image. I assumed it's `ozkary-credit:latest` based on the context.

- **`memory` and `cpu` values in `resources`**:
  - `requests`: These are the initial resource allocations for the container.
  - `limits`: These are the maximum allowed resource allocations for the container.

  Adjust the values based on your application's requirements and the resources available on your cluster. In the example, I set the initial memory to 64MiB and CPU to 100m (millicores), and the maximum limits to 128MiB and 200m, respectively. You can adjust these values based on your application's needs.

### Resource Specifications

  In the context of Kubernetes resource specifications, `Mi` and `m` are units used for memory and CPU resources, respectively. They are part of the resource quantity notation in Kubernetes.

1. **Memory (RAM):**
   - `Mi`: Megabytes (binary), equivalent to 2^20 bytes.
     - For example, 64Mi means 64 Megabytes.
   - `MB`: Megabytes (decimal), equivalent to 10^6 bytes.

   In Kubernetes, `Mi` is used to represent quantities in binary megabytes, which is the convention for memory sizes in the context of computers and operating systems.

2. **CPU:**
   - `m`: MilliCPU, or millicores. It represents a thousandth of a CPU core.
     - For example, 100m means 100 milliCPU, which is 0.1 CPU core.
   - `CPU cores`: Whole CPU cores, where 1 CPU core is equivalent to 1000m or 1000 millicores.

   The `m` notation is commonly used in Kubernetes to express fractional CPU resources, allowing for fine-grained control over CPU allocations.

In summary:
- `Mi` is used for memory and represents binary megabytes.
- `m` is used for CPU and represents millicores, which are a fraction of a CPU core.

When specifying resource limits and requests in Kubernetes YAML files, you can use these units to define how much CPU and memory your containers require or are allowed to use. Adjusting these values appropriately ensures efficient resource utilization and prevents resource contention in a Kubernetes cluster.

- **`containerPort`**: Replace `<Port>` with the actual port your application listens on. I assumed it's 9696 based on the context.

Make sure to replace placeholders like `<Image>` and `<Port>` with the actual values for your application. Adjust the resource values (`memory` and `cpu`) based on your application's requirements and the available resources on your cluster.

3. **Apply Kubernetes Manifests:**
   - Apply the updated Kubernetes manifests using `kubectl apply`:

   ```bash
   kubectl apply -f your-deployment-file.yaml
   ```

   Replace `your-deployment-file.yaml` with the path to your updated YAML file.

4. **Verify Deployment:**
   - Check the status of your deployment:

   ```bash
   kubectl get deployments
   ```

   Ensure that your deployment is running and the pods are using the Docker image you specified.

   The output from the kubectl get deployments command provides information about the status of your deployments in a Kubernetes cluster. Let's break down the columns in the output:

    - NAME: The name of the deployment. In your case, it's named "credit."

    - READY: This column shows the current status of the pods managed by the deployment. The format is current pods/total pods. In your case, it says 1/1, which means there is one pod running and it is ready.

    - UP-TO-DATE: This column indicates how many replicas of the desired deployment are running the latest version of the container image. In your case, it says 1, which means that the deployment is up-to-date with the desired configuration.

    - AVAILABLE: This column shows how many replicas of the application are available to your users. In your case, it says 1, indicating that one replica is available for serving traffic.

    - AGE: This column displays the age of the deployment, which is the time elapsed since it was created.

The `kind load docker-image` command simplifies the process by allowing you to load Docker images directly into `kind` without pushing them to a registry first.

Note: Ensure that your `kind` cluster is up and running (`kind create cluster`), and you have the necessary Kubernetes manifests to deploy your application. Adjust the steps based on your specific deployment requirements and configuration.

## Create a Service

In the `Service` YAML file, the `selector` field is used to target the pods to which the service should direct traffic. This value should match the labels applied to the pods you want to expose through the service. In our case, the deployment named "credit", so the `selector` should use the same labels that the "credit" deployment uses.

Assuming that your "credit" deployment uses the label `app: credit`, your `Service` YAML file should look like this:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-credit-service  # Specify the name for your service
spec:
  type: LoadBalancer
  selector:
    app: credit  # Use the same label as your deployment
  ports:
  - port: 80
    targetPort: <PORT>  # Replace <PORT> with the port your application is listening on
```

Make sure to replace `<PORT>` with the actual port your application inside the pods is configured to use.

So, if your "credit" deployment has labels like `app: credit`, the `selector` in your service should match those labels (`app: credit`). Also, specify a name for your service in the `metadata` section under `name`. In the example above, I used "my-credit-service" as a placeholder. You can choose a meaningful name for your service.

Remember to apply this service YAML using `kubectl apply -f your-service-file.yaml` after you've made the necessary changes.

## Test the service

The `kubectl port-forward` command creates a temporary port-forwarding session from your local machine to a specific pod within your Kubernetes cluster. The port-forwarding session remains active as long as the `kubectl port-forward` command is running in your terminal.

Here's how you typically use it:

```bash
kubectl port-forward service/ozkary-credit-service 8080:80
```

In this example:
- `service/ozkary-credit-service` specifies the service you want to forward to.
- `8080:80` specifies that traffic sent to your local port 8080 will be forwarded to port 80 on the specified service.

As long as this command is running in your terminal, the port-forwarding session is active, and you can access the service locally at `http://localhost:8080`.

If you want to keep the port-forwarding session running while you do other tasks, you can run the command in the background by appending `&` at the end:

```bash
kubectl port-forward service/ozkary-credit-service 8080:80 &
```

Keep in mind that this will keep the process running in the background until you manually stop it.

However, for long-term or production use cases, using `kubectl port-forward` may not be the ideal solution. Instead, consider using services of type `LoadBalancer` or `NodePort` for exposing services outside the cluster. These types of services provide a more robust and scalable solution, especially in a production environment.

## Auto Scaling

Auto-scaling in Kubernetes can be achieved using the Horizontal Pod Autoscaler (HPA). The HPA automatically adjusts the number of replica pods in a deployment or replica set based on observed metrics such as CPU utilization or custom metrics.

Here's a general guide on how to set up Horizontal Pod Autoscaling for your deployment:

1. **Ensure Metrics Server is Installed:**
   - The Horizontal Pod Autoscaler relies on the Metrics Server to collect resource utilization metrics. Make sure the Metrics Server is installed in your cluster. You can install it using the following command:

     ```bash
     kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
     ```

2. **Enable Metrics on Deployment:**
   - Ensure that your deployment has resource requests defined for the metrics you want to use for autoscaling. For example, if you want to scale based on CPU usage, make sure your deployment YAML includes CPU resource requests:

     ```yaml
     apiVersion: apps/v1
     kind: Deployment
     metadata:
       name: credit
     spec:
       replicas: 1
       template:
         metadata:
           labels:
             app: credit
         spec:
           containers:
           - name: credit
             image: ozkary-credit:latest
             resources:
               requests:
                 memory: "64Mi"
                 cpu: "100m"
               limits:
                 memory: "128Mi"
                 cpu: "200m"
           # ... (other deployment settings)
     ```

3. **Create Horizontal Pod Autoscaler (HPA):**
   - Create an HPA resource to define the autoscaling behavior. Below is an example for autoscaling based on CPU utilization:

     ```yaml
     apiVersion: autoscaling/v2beta2
     kind: HorizontalPodAutoscaler
     metadata:
       name: credit-autoscaler
     spec:
       scaleTargetRef:
         apiVersion: apps/v1
         kind: Deployment
         name: credit
       minReplicas: 1
       maxReplicas: 5
       metrics:
       - type: Resource
         resource:
           name: cpu
           targetAverageUtilization: 50
     ```

     In this example, the autoscaler targets the deployment named "credit," with a target average CPU utilization of 50%. Adjust the `minReplicas` and `maxReplicas` values based on your scaling requirements.

4. **Apply the HPA:**
   - Apply the HPA to your cluster:

     ```bash
     kubectl apply -f your-hpa-file.yaml
     ```
5. **Check the Status:**
   
    ```bash
    kubectl get hpa

    NAME         REFERENCE           TARGETS         MINPODS   MAXPODS   REPLICAS   AGE
    credit-hpa   Deployment/credit   <unknown>/20%   1         3         0          9s
    ```


Now, the HPA will automatically adjust the number of replicas for your "credit" deployment based on CPU utilization. You can customize the HPA configuration based on your specific scaling needs and metrics. Adjust the `metrics` section and other parameters accordingly.

Keep in mind that the metrics used for autoscaling depend on the type of workload and your specific requirements. You might use CPU utilization, memory usage, or custom metrics based on your application's behavior.

## Stop the service

To stop a Kubernetes service, you typically don't "stop" it directly; instead, you delete the resources associated with the service. In your case, you want to delete the `Service` resource you created.

Here's how you can do it:

```bash
kubectl delete service ozkary-credit-service
```

This command deletes the service named "ozkary-credit-service." After running this command, the external IP address (if provisioned) and the service itself will be removed.

If you also want to stop the associated `Deployment`, you can delete it as well:

```bash
kubectl delete deployment credit
```

Replace "credit" with the name of your deployment. This command deletes the deployment, and as a result, the associated pods will be terminated.

If you want to delete both the `Service` and the `Deployment` in one command, you can use the `kubectl delete` command with multiple resource types:

```bash
kubectl delete service,deployment ozkary-credit-service credit
```

This deletes both the service and deployment in one command.

Remember that deleting resources is irreversible, so make sure you want to stop and remove these components from your cluster. If you need to recreate the service or deployment later, you can do so by applying the respective YAML files again using `kubectl apply`.

## Delete Resources

When you're finished working with your Kubernetes cluster and services, you might want to clean up resources to ensure that no unnecessary components are running. Here are a few additional commands you can use:

1. **Delete Kubernetes Resources:**
   - If you want to delete all resources in a specific namespace (e.g., the default namespace), you can use the following command:

     ```bash
     kubectl delete all --all
     ```

   - This deletes all resources of types pods, services, deployments, etc., in the specified namespace.

2. **Delete the Kind Cluster:**
   - If you created a cluster using `kind` and you want to delete the entire cluster, you can use:

     ```bash
     kind delete cluster
     ```

   - This command removes the entire `kind` cluster.

3. **Stop `kubectl port-forward`:**
   - If you have any active `kubectl port-forward` sessions, you might want to stop them by finding the corresponding processes and terminating them. For example:

     ```bash
     pkill -f 'kubectl port-forward'
     ```

   - This command finds and terminates any processes related to `kubectl port-forward`.

Remember to use these commands cautiously, especially when using `kubectl delete all --all`, as it will delete all resources in the specified namespace.

Additionally, if you are using `kind`, you might want to check if there are any remaining Docker containers or images related to your `kind` cluster. You can use Docker commands like `docker ps` and `docker images` to list and remove containers and images.

Always be mindful when deleting resources, especially in a production environment, to avoid unintentional data loss.